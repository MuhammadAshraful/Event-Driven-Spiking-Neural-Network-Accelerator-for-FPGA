"""Unsupervised 8x8 MNIST SNN classifier reference.

This is the large-scale reference for the custom RTL classifier.  Training is
fully unsupervised: labels are loaded only after training to assign each output
neuron to the digit it won most often.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_RAW = REPO_ROOT / "software" / "python" / "data" / "MNIST" / "raw"
OUT_DIR = REPO_ROOT / "outputs"
SIM_DIR = REPO_ROOT / "hardware" / "sim_work_rtl_mnist_classifier"

INPUT_NEURONS = 64
OUTPUT_NEURONS = 10
CLK_HZ = 100_000_000


@dataclass
class ClassifierMetrics:
    train_images: int
    test_images: int
    input_neurons: int
    output_neurons: int
    encoding_method: str
    stdp_parameters: Dict[str, float]
    wta_parameters: Dict[str, float]
    accuracy: float
    confusion_matrix: List[List[int]]
    per_class_accuracy: Dict[str, float]
    assigned_labels: List[int]
    train_winner_counts: List[int]
    average_input_spikes_per_image: float
    average_output_spikes_per_image: float
    average_latency_cycles: float
    average_latency_us_100mhz: float
    throughput_images_per_second: float
    weight_summary: Dict[str, float]
    rtl_comparison: Dict[str, object]


def load_idx_images(path: Path, count: int) -> np.ndarray:
    data = path.read_bytes()
    magic, available, rows, cols = struct.unpack_from(">IIII", data, 0)
    if magic != 2051:
        raise ValueError(f"{path} is not an IDX image file")
    count = min(count, available)
    arr = np.frombuffer(data, dtype=np.uint8, offset=16, count=count * rows * cols)
    return arr.reshape(count, rows, cols).astype(np.float32)


def load_idx_labels(path: Path, count: int) -> np.ndarray:
    data = path.read_bytes()
    magic, available = struct.unpack_from(">II", data, 0)
    if magic != 2049:
        raise ValueError(f"{path} is not an IDX label file")
    count = min(count, available)
    return np.frombuffer(data, dtype=np.uint8, offset=8, count=count).astype(np.int64)


def downsample_8x8(images: np.ndarray) -> np.ndarray:
    pooled = np.zeros((images.shape[0], 8, 8), dtype=np.float32)
    for out_y in range(8):
        y0 = int(round(out_y * 28 / 8))
        y1 = int(round((out_y + 1) * 28 / 8))
        for out_x in range(8):
            x0 = int(round(out_x * 28 / 8))
            x1 = int(round((out_x + 1) * 28 / 8))
            pooled[:, out_y, out_x] = images[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return (pooled.reshape(images.shape[0], 64) / 255.0).clip(0.0, 1.0)


def normalized_inputs(x: np.ndarray) -> np.ndarray:
    # Mild contrast normalization helps 8x8 prototypes learn stroke shape rather
    # than raw brightness while staying compatible with simple RTL event weights.
    denom = np.maximum(x.max(axis=1, keepdims=True), 1e-6)
    return (x / denom).clip(0.0, 1.0)


def make_initial_weights(rng: np.random.Generator, train_x: np.ndarray, target_weight_sum: float) -> np.ndarray:
    seed_indices = rng.choice(train_x.shape[0], size=OUTPUT_NEURONS, replace=False)
    weights = train_x[seed_indices].astype(np.float32).copy()
    weights += rng.uniform(0.0, 0.03, size=weights.shape).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    weights *= target_weight_sum
    return weights


def choose_winner(x: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> Tuple[int, np.ndarray]:
    membrane = weights @ x
    score = membrane - thresholds
    winner = int(np.argmax(score))
    return winner, membrane


def train_unsupervised(
    train_x: np.ndarray,
    *,
    epochs: int,
    seed: int,
    learning_rate: float,
    lr_decay: float,
    threshold_inc: float,
    threshold_decay: float,
    target_weight_sum: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    weights = make_initial_weights(rng, train_x, target_weight_sum)
    initial_weights = weights.copy()
    thresholds = np.zeros(OUTPUT_NEURONS, dtype=np.float32)
    winner_counts = np.zeros(OUTPUT_NEURONS, dtype=np.int64)
    latency_cycles: List[int] = []

    indices = np.arange(train_x.shape[0])
    for epoch in range(epochs):
        rng.shuffle(indices)
        lr = learning_rate * (lr_decay ** epoch)
        for idx in indices:
            x = train_x[idx]
            winner, membrane = choose_winner(x, weights, thresholds)
            winner_counts[winner] += 1

            # Pair-based STDP-inspired competitive update:
            # active pre + winning post strengthens matching inputs; inactive
            # inputs depress. This is equivalent to moving the winner prototype
            # toward the presented spike pattern.
            weights[winner] += lr * (x * target_weight_sum / max(1.0, x.sum()) - weights[winner])
            weights[winner] = np.clip(weights[winner], 0.0, 1.0)
            weights[winner] *= target_weight_sum / max(target_weight_sum, weights[winner].sum()) if weights[winner].sum() > target_weight_sum else 1.0

            thresholds *= (1.0 - threshold_decay)
            thresholds[winner] += threshold_inc

            active = int(np.count_nonzero(x > 0.05))
            latency_cycles.append(max(1, active))

    return weights, thresholds, winner_counts, initial_weights, float(np.mean(latency_cycles))


def assign_labels(train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> Tuple[List[int], np.ndarray]:
    counts = np.zeros((OUTPUT_NEURONS, 10), dtype=np.int64)
    winners = np.zeros(train_x.shape[0], dtype=np.int64)
    eval_thresholds = np.zeros_like(thresholds)
    for idx, x in enumerate(train_x):
        winner, _ = choose_winner(x, weights, eval_thresholds)
        winners[idx] = winner
        counts[winner, int(train_y[idx])] += 1

    global_majority = int(np.bincount(train_y, minlength=10).argmax())
    assigned = []
    for out_idx in range(OUTPUT_NEURONS):
        if counts[out_idx].sum() == 0:
            assigned.append(global_majority)
        else:
            assigned.append(int(counts[out_idx].argmax()))
    return assigned, winners


def evaluate(
    test_x: np.ndarray,
    test_y: np.ndarray,
    weights: np.ndarray,
    assigned_labels: Sequence[int],
) -> Tuple[float, List[List[int]], Dict[str, float], np.ndarray, float]:
    confusion = np.zeros((10, 10), dtype=np.int64)
    winner_counts = np.zeros(OUTPUT_NEURONS, dtype=np.int64)
    latencies: List[int] = []
    correct = 0
    thresholds = np.zeros(OUTPUT_NEURONS, dtype=np.float32)

    for x, label in zip(test_x, test_y):
        winner, _ = choose_winner(x, weights, thresholds)
        winner_counts[winner] += 1
        pred = int(assigned_labels[winner])
        confusion[int(label), pred] += 1
        correct += int(pred == int(label))
        latencies.append(max(1, int(np.count_nonzero(x > 0.05))))

    per_class: Dict[str, float] = {}
    for digit in range(10):
        total = int(confusion[digit].sum())
        per_class[str(digit)] = float(confusion[digit, digit] / total) if total else 0.0
    return float(correct / len(test_y)), confusion.tolist(), per_class, winner_counts, float(np.mean(latencies))


def parse_rtl_logs() -> Dict[str, object]:
    results: Dict[str, object] = {}
    pattern = re.compile(
        r"MNIST_CLASSIFIER_RTL_SUMMARY subset=(\S+) train_images=(\d+) test_images=(\d+) "
        r"assigned_accuracy_permille=(\d+) avg_latency_cycles=(\d+) avg_output_spikes_x1000=(\d+)"
    )
    for log in SIM_DIR.glob("sim_tb_custom_rtl_mnist_classifier_*.log"):
        text = log.read_text(encoding="ascii", errors="ignore")
        match = pattern.search(text)
        if match:
            subset = match.group(1)
            latency_cycles = int(match.group(5))
            results[subset] = {
                "log": str(log),
                "train_images": int(match.group(2)),
                "test_images": int(match.group(3)),
                "accuracy": int(match.group(4)) / 1000.0,
                "avg_latency_cycles": latency_cycles,
                "avg_latency_us_100mhz": latency_cycles / (CLK_HZ / 1_000_000),
                "throughput_images_per_second_100mhz": CLK_HZ / max(1, latency_cycles),
                "avg_output_spikes_per_image": int(match.group(6)) / 1000.0,
                "status": "PASS" if "MNIST CLASSIFIER RTL TEST PASSED" in text else "FAIL",
            }
    return results


def write_outputs(metrics: ClassifierMetrics) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "mnist_classifier_results.json"
    md_path = OUT_DIR / "mnist_classifier_results.md"
    json_path.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="ascii")

    lines = [
        "# MNIST Classifier Results",
        "",
        f"- Train images: {metrics.train_images}",
        f"- Test images: {metrics.test_images}",
        f"- Input neurons: {metrics.input_neurons}",
        f"- Output neurons: {metrics.output_neurons}",
        f"- Encoding: {metrics.encoding_method}",
        f"- Accuracy: {metrics.accuracy:.4f}",
        f"- Average input spikes/image: {metrics.average_input_spikes_per_image:.2f}",
        f"- Average output spikes/image: {metrics.average_output_spikes_per_image:.2f}",
        f"- Average latency: {metrics.average_latency_cycles:.2f} cycles ({metrics.average_latency_us_100mhz:.3f} us at 100 MHz)",
        f"- Throughput estimate: {metrics.throughput_images_per_second:.2f} images/s",
        f"- Assigned labels per output neuron: {metrics.assigned_labels}",
        "",
        "## Weight Summary",
        "",
    ]
    for key, value in metrics.weight_summary.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(["", "## Per-Class Accuracy", ""])
    for digit, acc in metrics.per_class_accuracy.items():
        lines.append(f"- {digit}: {acc:.4f}")
    lines.extend(["", "## Confusion Matrix", "", "Rows=true labels, columns=predicted labels.", ""])
    lines.append("| true\\pred | " + " | ".join(str(i) for i in range(10)) + " |")
    lines.append("|---|" + "|".join(["---:"] * 10) + "|")
    for idx, row in enumerate(metrics.confusion_matrix):
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    lines.extend(["", "## RTL Comparison", ""])
    if metrics.rtl_comparison:
        for subset, result in metrics.rtl_comparison.items():
            lines.append(f"- {subset}: {result}")
    else:
        lines.append("- No RTL classifier logs found yet.")
    md_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.22)
    parser.add_argument("--lr-decay", type=float, default=0.88)
    parser.add_argument("--threshold-inc", type=float, default=0.035)
    parser.add_argument("--threshold-decay", type=float, default=0.003)
    parser.add_argument("--target-weight-sum", type=float, default=18.0)
    args = parser.parse_args()

    train_images = load_idx_images(MNIST_RAW / "train-images-idx3-ubyte", args.train_count)
    train_labels = load_idx_labels(MNIST_RAW / "train-labels-idx1-ubyte", args.train_count)
    test_images = load_idx_images(MNIST_RAW / "t10k-images-idx3-ubyte", args.test_count)
    test_labels = load_idx_labels(MNIST_RAW / "t10k-labels-idx1-ubyte", args.test_count)
    train_x = normalized_inputs(downsample_8x8(train_images))
    test_x = normalized_inputs(downsample_8x8(test_images))

    weights, thresholds, train_win_counts, initial_weights, train_latency = train_unsupervised(
        train_x,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        threshold_inc=args.threshold_inc,
        threshold_decay=args.threshold_decay,
        target_weight_sum=args.target_weight_sum,
    )
    assigned_labels, train_winners = assign_labels(train_x, train_labels, weights, thresholds)
    accuracy, confusion, per_class, test_win_counts, test_latency = evaluate(test_x, test_labels, weights, assigned_labels)

    avg_input_spikes = float(np.count_nonzero(test_x > 0.05, axis=1).mean())
    avg_output_spikes = 1.0
    avg_latency = test_latency
    throughput = CLK_HZ / max(1.0, avg_latency)
    weight_delta = weights - initial_weights

    metrics = ClassifierMetrics(
        train_images=int(train_x.shape[0]),
        test_images=int(test_x.shape[0]),
        input_neurons=INPUT_NEURONS,
        output_neurons=OUTPUT_NEURONS,
        encoding_method="28x28 MNIST average-pooled to 8x8, contrast-normalized rate vector",
        stdp_parameters={
            "learning_rate": args.learning_rate,
            "lr_decay": args.lr_decay,
            "epochs": args.epochs,
            "target_weight_sum": args.target_weight_sum,
            "rule": "winner-only competitive LTP/LTD prototype update",
        },
        wta_parameters={
            "outputs": OUTPUT_NEURONS,
            "winner": "argmax membrane minus adaptive threshold",
            "threshold_inc": args.threshold_inc,
            "threshold_decay": args.threshold_decay,
        },
        accuracy=accuracy,
        confusion_matrix=confusion,
        per_class_accuracy=per_class,
        assigned_labels=assigned_labels,
        train_winner_counts=[int(v) for v in np.bincount(train_winners, minlength=OUTPUT_NEURONS)],
        average_input_spikes_per_image=avg_input_spikes,
        average_output_spikes_per_image=avg_output_spikes,
        average_latency_cycles=avg_latency,
        average_latency_us_100mhz=avg_latency / (CLK_HZ / 1_000_000),
        throughput_images_per_second=throughput,
        weight_summary={
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "mean_abs_delta": float(np.abs(weight_delta).mean()),
            "max_abs_delta": float(np.abs(weight_delta).max()),
        },
        rtl_comparison=parse_rtl_logs(),
    )

    write_outputs(metrics)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Assigned labels: {assigned_labels}")
    print(f"Train winner counts: {metrics.train_winner_counts}")
    print(f"Average latency cycles: {avg_latency:.2f}")
    print(f"Throughput estimate: {throughput:.2f} images/s")
    print(f"Wrote {OUT_DIR / 'mnist_classifier_results.md'}")
    print(f"Wrote {OUT_DIR / 'mnist_classifier_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
