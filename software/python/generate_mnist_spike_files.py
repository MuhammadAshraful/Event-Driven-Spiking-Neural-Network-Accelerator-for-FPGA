"""Generate tiny MNIST spike-event files for RTL unsupervised STDP smoke tests.

The RTL smoke topology has three input neurons, so each 28x28 MNIST image is
pooled into three vertical bands. Labels are saved only for post-training
evaluation; the RTL testbench does not read labels.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_RAW = REPO_ROOT / "software" / "python" / "data" / "MNIST" / "raw"
OUT_DIR = REPO_ROOT / "hardware" / "hdl" / "tb" / "data"


@dataclass(frozen=True)
class SpikeEvent:
    cycle: int
    neuron_id: int
    weight: int


def load_idx_images(path: Path, count: int) -> List[List[int]]:
    data = path.read_bytes()
    magic, num_images, rows, cols = struct.unpack_from(">IIII", data, 0)
    if magic != 2051:
        raise ValueError(f"{path} is not an IDX image file")
    if rows != 28 or cols != 28:
        raise ValueError(f"expected 28x28 MNIST images, got {rows}x{cols}")

    count = min(count, num_images)
    images: List[List[int]] = []
    offset = 16
    image_size = rows * cols
    for idx in range(count):
        start = offset + idx * image_size
        images.append(list(data[start : start + image_size]))
    return images


def load_idx_labels(path: Path, count: int) -> List[int]:
    data = path.read_bytes()
    magic, num_labels = struct.unpack_from(">II", data, 0)
    if magic != 2049:
        raise ValueError(f"{path} is not an IDX label file")
    count = min(count, num_labels)
    return list(data[8 : 8 + count])


def pooled_band_spike_counts(image: Sequence[int], max_spikes_per_band: int) -> List[int]:
    bands = [(0, 9), (9, 19), (19, 28)]
    counts: List[int] = []
    totals: List[int] = []
    for col_start, col_stop in bands:
        total = 0
        for row in range(28):
            for col in range(col_start, col_stop):
                total += image[row * 28 + col]

        totals.append(total)
        count = int(round(total / (255.0 * 12.0)))
        if total >= 255 * 4:
            count = max(1, count)
        counts.append(min(max_spikes_per_band, count))

    # Keep the RTL smoke alive even for sparse digits. This does not use labels;
    # it only ensures at least one active input channel when the image is nonzero.
    if sum(counts) == 0 and max(image) > 0:
        strongest = max(range(3), key=lambda idx: totals[idx])
        counts[strongest] = 1
    return counts


def encode_images(
    images: Sequence[Sequence[int]],
    *,
    image_gap: int,
    start_cycle: int,
    event_spacing: int,
    spike_weight: int,
    max_spikes_per_band: int,
) -> List[SpikeEvent]:
    events: List[SpikeEvent] = []
    for image_idx, image in enumerate(images):
        counts = pooled_band_spike_counts(image, max_spikes_per_band)
        cycle = image_idx * image_gap + start_cycle
        for burst in range(max_spikes_per_band):
            for neuron_id, count in enumerate(counts):
                if count > burst:
                    events.append(SpikeEvent(cycle, neuron_id, spike_weight))
                    cycle += event_spacing
    return events


def write_events(path: Path, events: Iterable[SpikeEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        for event in events:
            fh.write(f"{event.cycle} {event.neuron_id} {event.weight}\n")


def write_labels(path: Path, labels_by_file: Sequence[Tuple[str, Sequence[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        fh.write("file image_index label\n")
        for filename, labels in labels_by_file:
            for idx, label in enumerate(labels):
                fh.write(f"{filename} {idx} {label}\n")


def generate_one(name: str, count: int, args: argparse.Namespace) -> Tuple[str, List[int], int]:
    images = load_idx_images(MNIST_RAW / "train-images-idx3-ubyte", count)
    labels = load_idx_labels(MNIST_RAW / "train-labels-idx1-ubyte", count)
    events = encode_images(
        images,
        image_gap=args.image_gap,
        start_cycle=args.start_cycle,
        event_spacing=args.event_spacing,
        spike_weight=args.spike_weight,
        max_spikes_per_band=args.max_spikes_per_band,
    )
    filename = f"mnist_unsup_{name}.mem"
    write_events(OUT_DIR / filename, events)
    print(f"{filename}: {len(events)} spikes from {count} image(s)")
    return filename, labels, len(events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", type=int, default=1, help="images for mnist_unsup_1.mem")
    parser.add_argument("--ten", type=int, default=10, help="images for mnist_unsup_10.mem")
    parser.add_argument("--batch", type=int, default=25, help="images for mnist_unsup_batch.mem")
    parser.add_argument("--image-gap", type=int, default=250)
    parser.add_argument("--start-cycle", type=int, default=10)
    parser.add_argument("--event-spacing", type=int, default=6)
    parser.add_argument("--spike-weight", type=int, default=12)
    parser.add_argument("--max-spikes-per-band", type=int, default=8)
    args = parser.parse_args()

    generated = [
        generate_one("1", args.one, args),
        generate_one("10", args.ten, args),
        generate_one("batch", args.batch, args),
    ]
    write_labels(OUT_DIR / "mnist_unsup_labels.txt", [(name, labels) for name, labels, _ in generated])
    print(f"labels: {OUT_DIR / 'mnist_unsup_labels.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
