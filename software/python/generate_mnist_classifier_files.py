"""Generate 8x8 MNIST event files for the RTL classifier workflow.

Format:
    cycle neuron_id weight image_id phase

phase is 0 for training and 1 for inference/test. Labels are written to
separate files and are not consumed by the RTL training loop.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_RAW = REPO_ROOT / "software" / "python" / "data" / "MNIST" / "raw"
OUT_DIR = REPO_ROOT / "hardware" / "hdl" / "tb" / "data"


def load_idx_images(path: Path, count: int) -> List[List[int]]:
    data = path.read_bytes()
    magic, available, rows, cols = struct.unpack_from(">IIII", data, 0)
    if magic != 2051:
        raise ValueError(f"{path} is not an IDX image file")
    count = min(count, available)
    image_size = rows * cols
    images: List[List[int]] = []
    for idx in range(count):
        start = 16 + idx * image_size
        images.append(list(data[start : start + image_size]))
    return images


def load_idx_labels(path: Path, count: int) -> List[int]:
    data = path.read_bytes()
    magic, available = struct.unpack_from(">II", data, 0)
    if magic != 2049:
        raise ValueError(f"{path} is not an IDX label file")
    count = min(count, available)
    return list(data[8 : 8 + count])


def downsample_8x8(image: Sequence[int]) -> List[int]:
    """Average-pool 28x28 to 8x8 and quantize to 0..15."""
    values: List[int] = []
    for out_y in range(8):
        y0 = int(round(out_y * 28 / 8))
        y1 = int(round((out_y + 1) * 28 / 8))
        for out_x in range(8):
            x0 = int(round(out_x * 28 / 8))
            x1 = int(round((out_x + 1) * 28 / 8))
            total = 0
            samples = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    total += image[y * 28 + x]
                    samples += 1
            avg = total / max(1, samples)
            values.append(int(round(avg * 15 / 255)))
    return values


def events_for_images(
    images: Sequence[Sequence[int]],
    *,
    phase: int,
    image_gap: int,
    start_cycle: int,
    event_spacing: int,
    min_weight: int,
) -> List[Tuple[int, int, int, int, int]]:
    events: List[Tuple[int, int, int, int, int]] = []
    for image_id, image in enumerate(images):
        pooled = downsample_8x8(image)
        cycle = image_id * image_gap + start_cycle
        for neuron_id, weight in enumerate(pooled):
            if weight >= min_weight:
                events.append((cycle, neuron_id, weight, image_id, phase))
                cycle += event_spacing
    return events


def write_events(path: Path, events: Iterable[Tuple[int, int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        for cycle, neuron_id, weight, image_id, phase in events:
            fh.write(f"{cycle} {neuron_id} {weight} {image_id} {phase}\n")


def write_labels(path: Path, labels: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        fh.write("image_id label\n")
        for image_id, label in enumerate(labels):
            fh.write(f"{image_id} {label}\n")


def write_vectors(path: Path, images: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        for image_id, image in enumerate(images):
            pooled = downsample_8x8(image)
            fh.write(f"{image_id} " + " ".join(str(v) for v in pooled) + "\n")


def generate_subset(name: str, train_count: int, test_count: int, args: argparse.Namespace) -> None:
    train_images = load_idx_images(MNIST_RAW / "train-images-idx3-ubyte", train_count)
    train_labels = load_idx_labels(MNIST_RAW / "train-labels-idx1-ubyte", train_count)
    test_images = load_idx_images(MNIST_RAW / "t10k-images-idx3-ubyte", test_count)
    test_labels = load_idx_labels(MNIST_RAW / "t10k-labels-idx1-ubyte", test_count)

    train_events = events_for_images(
        train_images,
        phase=0,
        image_gap=args.image_gap,
        start_cycle=args.start_cycle,
        event_spacing=args.event_spacing,
        min_weight=args.min_weight,
    )
    test_events = events_for_images(
        test_images,
        phase=1,
        image_gap=args.image_gap,
        start_cycle=args.start_cycle,
        event_spacing=args.event_spacing,
        min_weight=args.min_weight,
    )

    write_events(OUT_DIR / f"mnist_classifier_train_{name}.mem", train_events)
    write_events(OUT_DIR / f"mnist_classifier_test_{name}.mem", test_events)
    write_labels(OUT_DIR / f"mnist_classifier_train_{name}_labels.txt", train_labels)
    write_labels(OUT_DIR / f"mnist_classifier_test_{name}_labels.txt", test_labels)
    write_vectors(OUT_DIR / f"mnist_classifier_train_{name}_vectors.txt", train_images)
    write_vectors(OUT_DIR / f"mnist_classifier_test_{name}_vectors.txt", test_images)

    print(
        f"{name}: train_images={len(train_images)} train_events={len(train_events)} "
        f"test_images={len(test_images)} test_events={len(test_events)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--image-gap", type=int, default=140)
    parser.add_argument("--start-cycle", type=int, default=8)
    parser.add_argument("--event-spacing", type=int, default=1)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument("--subsets", nargs="*", default=["10", "100", "1000"])
    args = parser.parse_args()

    counts = {"10": 10, "100": 100, "1000": 1000}
    for name in args.subsets:
        if name not in counts:
            raise ValueError(f"unknown subset {name}; use one of {sorted(counts)}")
        test_count = min(args.test_count, counts[name])
        generate_subset(name, counts[name], test_count, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
