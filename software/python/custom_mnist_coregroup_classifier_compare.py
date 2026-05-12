"""Python comparison for the one-core_group MNIST classifier path.

This mirrors the simplified RTL core_group classifier at the image-window level:
integer weights, LIF-like output membrane accumulation, WTA, saturated STDP
updates, and post-training label assignment.  It also parses representative RTL
xsim logs when available.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "hardware" / "hdl" / "tb" / "data"
SIM_DIR = REPO_ROOT / "hardware" / "sim_work_rtl_mnist_coregroup_classifier"
OUT_DIR = REPO_ROOT / "outputs"

INPUT_NEURONS = 64
OUTPUT_NEURONS = 10
THRESHOLD = 8
W_MIN = 0
W_MAX = 15
A_PLUS = 2
A_MINUS = 1
CLK_HZ = 100_000_000


@dataclass
class ImageEvents:
    weights: List[int]
    input_spikes: int


@dataclass
class ClassifierRun:
    train_images: int
    test_images: int
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
    learned_weight_updates: int
    weight_summary: Dict[str, float]


def initial_weight(out_idx: int, in_idx: int) -> int:
    pattern = (out_idx * 13 + in_idx * 7 + (in_idx >> 3) * 5 + (in_idx & 7) * 3) % 10
    return 4 + pattern


def read_events(path: Path) -> List[ImageEvents]:
    by_image: Dict[int, List[int]] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        _cycle, neuron_id, weight, image_id, _phase = (int(part) for part in line.split())
        if image_id not in by_image:
            by_image[image_id] = [0] * INPUT_NEURONS
        by_image[image_id][neuron_id] = min(15, by_image[image_id][neuron_id] + weight)
    return [
        ImageEvents(weights=by_image[idx], input_spikes=sum(1 for w in by_image[idx] if w > 0))
        for idx in sorted(by_image)
    ]


def read_labels(path: Path) -> List[int]:
    labels: List[int] = []
    for line in path.read_text(encoding="ascii").splitlines()[1:]:
        if not line.strip():
            continue
        image_id, label = (int(part) for part in line.split())
        while len(labels) <= image_id:
            labels.append(0)
        labels[image_id] = label
    return labels


def make_weights() -> List[List[int]]:
    return [[initial_weight(out_idx, in_idx) for in_idx in range(INPUT_NEURONS)] for out_idx in range(OUTPUT_NEURONS)]


def run_image(image: ImageEvents, weights: Sequence[Sequence[int]]) -> Tuple[int, bool, int, int]:
    membrane = [0] * OUTPUT_NEURONS
    refractory = [0] * OUTPUT_NEURONS
    first_winner = 0
    have_winner = False
    out_spikes = 0
    latency = 0
    cycle = 0

    for in_idx, active_weight in enumerate(image.weights):
        if active_weight <= 0:
            continue
        cycle += 1
        for out_idx in range(OUTPUT_NEURONS):
            if refractory[out_idx] > 0:
                refractory[out_idx] -= 1
                continue
            membrane[out_idx] += int(weights[out_idx][in_idx])
            if membrane[out_idx] >= THRESHOLD:
                out_spikes += 1
                membrane[out_idx] = 0
                refractory[out_idx] = 8
                if not have_winner:
                    first_winner = out_idx
                    have_winner = True
                    latency = cycle
    return first_winner, have_winner, latency if have_winner else cycle, out_spikes


def apply_stdp(image: ImageEvents, weights: List[List[int]], winner: int) -> int:
    updates = 0
    for in_idx, trace in enumerate(image.weights):
        old = weights[winner][in_idx]
        if trace > 0:
            new = min(W_MAX, old + A_PLUS + (trace >> 3))
        else:
            new = max(W_MIN, old - A_MINUS)
        if new != old:
            updates += 1
        weights[winner][in_idx] = new
    return updates


def train_assign_test(
    train_events: List[ImageEvents],
    train_labels: Sequence[int],
    test_events: List[ImageEvents],
    test_labels: Sequence[int],
) -> ClassifierRun:
    weights = make_weights()
    updates = 0

    for image in train_events:
        winner, have_winner, _latency, _out_spikes = run_image(image, weights)
        if have_winner:
            updates += apply_stdp(image, weights, winner)

    assign_counts = [[0 for _ in range(10)] for _ in range(OUTPUT_NEURONS)]
    train_winner_counts = [0 for _ in range(OUTPUT_NEURONS)]
    for image, label in zip(train_events, train_labels):
        winner, have_winner, _latency, _out_spikes = run_image(image, weights)
        if have_winner:
            assign_counts[winner][label] += 1
            train_winner_counts[winner] += 1

    assigned_labels = []
    for out_idx in range(OUTPUT_NEURONS):
        if sum(assign_counts[out_idx]) == 0:
            assigned_labels.append(0)
        else:
            assigned_labels.append(max(range(10), key=lambda digit: assign_counts[out_idx][digit]))

    confusion = [[0 for _ in range(10)] for _ in range(10)]
    correct = 0
    latency_sum = 0
    out_spike_sum = 0
    for image, label in zip(test_events, test_labels):
        winner, have_winner, latency, out_spikes = run_image(image, weights)
        pred = assigned_labels[winner] if have_winner else 0
        confusion[label][pred] += 1
        correct += int(pred == label)
        latency_sum += latency
        out_spike_sum += out_spikes

    per_class: Dict[str, float] = {}
    for digit in range(10):
        total = sum(confusion[digit])
        per_class[str(digit)] = confusion[digit][digit] / total if total else 0.0

    flat = [w for row in weights for w in row]
    avg_latency = latency_sum / max(1, len(test_events))
    return ClassifierRun(
        train_images=len(train_events),
        test_images=len(test_events),
        accuracy=correct / max(1, len(test_events)),
        confusion_matrix=confusion,
        per_class_accuracy=per_class,
        assigned_labels=assigned_labels,
        train_winner_counts=train_winner_counts,
        average_input_spikes_per_image=sum(img.input_spikes for img in test_events) / max(1, len(test_events)),
        average_output_spikes_per_image=out_spike_sum / max(1, len(test_events)),
        average_latency_cycles=avg_latency,
        average_latency_us_100mhz=avg_latency / (CLK_HZ / 1_000_000),
        throughput_images_per_second=CLK_HZ / max(1.0, avg_latency),
        learned_weight_updates=updates,
        weight_summary={
            "min": min(flat),
            "max": max(flat),
            "mean": sum(flat) / len(flat),
            "sum": sum(flat),
        },
    )


def parse_rtl_logs() -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    pattern = re.compile(
        r"MNIST_COREGROUP_RTL_SUMMARY subset=(\S+) train_images=(\d+) test_images=(\d+) "
        r"accuracy_permille=(\d+) avg_latency_cycles=(\d+) avg_output_spikes_x1000=(\d+) "
        r"avg_input_spikes_x1000=(\d+) throughput_images_per_sec=(\d+) "
        r"weight_min=(\d+) weight_max=(\d+) weight_sum=(\d+) total_updates=(\d+) sim_cycles=(\d+)"
    )
    for log in SIM_DIR.glob("sim_tb_custom_rtl_mnist_coregroup_classifier_*.log"):
        text = log.read_text(encoding="ascii", errors="ignore")
        match = pattern.search(text)
        if not match:
            continue
        subset = match.group(1)
        results[subset] = {
            "log": str(log),
            "train_images": int(match.group(2)),
            "test_images": int(match.group(3)),
            "accuracy": int(match.group(4)) / 1000.0,
            "avg_latency_cycles": int(match.group(5)),
            "avg_latency_us_100mhz": int(match.group(5)) / (CLK_HZ / 1_000_000),
            "avg_output_spikes_per_image": int(match.group(6)) / 1000.0,
            "avg_input_spikes_per_image": int(match.group(7)) / 1000.0,
            "throughput_images_per_second": int(match.group(8)),
            "weight_min": int(match.group(9)),
            "weight_max": int(match.group(10)),
            "weight_sum": int(match.group(11)),
            "learned_weight_updates": int(match.group(12)),
            "sim_cycles": int(match.group(13)),
            "status": "PASS" if "MNIST COREGROUP CLASSIFIER RTL TEST PASSED" in text else "FAIL",
        }
    return results


def write_outputs(py_result: ClassifierRun, rtl_results: Dict[str, Dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "python_reference": asdict(py_result),
        "rtl_results": rtl_results,
        "python_comparison_status": {
            subset: "PASS" if result.get("status") == "PASS" and result.get("learned_weight_updates", 0) > 0 else "FAIL"
            for subset, result in rtl_results.items()
        },
    }
    (OUT_DIR / "mnist_coregroup_classifier_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )

    lines = [
        "# MNIST Coregroup Classifier Results",
        "",
        "## Python Reference",
        "",
        f"- Train images: {py_result.train_images}",
        f"- Test images: {py_result.test_images}",
        f"- Input neurons: {INPUT_NEURONS}",
        f"- Output neurons: {OUTPUT_NEURONS}",
        f"- Accuracy: {py_result.accuracy:.4f}",
        f"- Average input spikes/image: {py_result.average_input_spikes_per_image:.2f}",
        f"- Average output spikes/image: {py_result.average_output_spikes_per_image:.2f}",
        f"- Average latency cycles: {py_result.average_latency_cycles:.2f}",
        f"- Average latency at 100 MHz: {py_result.average_latency_us_100mhz:.3f} us",
        f"- Throughput estimate: {py_result.throughput_images_per_second:.2f} images/s",
        f"- Learned weight updates: {py_result.learned_weight_updates}",
        f"- Assigned labels: {py_result.assigned_labels}",
        "",
        "## RTL Representative Runs",
        "",
        "| subset | train | test | accuracy | avg latency cycles | latency us | throughput img/s | avg input spikes | avg output spikes | updates | weights | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for subset in sorted(rtl_results, key=lambda s: int(s)):
        result = rtl_results[subset]
        lines.append(
            f"| {subset} | {result['train_images']} | {result['test_images']} | "
            f"{result['accuracy']:.4f} | {result['avg_latency_cycles']} | "
            f"{result['avg_latency_us_100mhz']:.3f} | {result['throughput_images_per_second']} | "
            f"{result['avg_input_spikes_per_image']:.2f} | {result['avg_output_spikes_per_image']:.2f} | "
            f"{result['learned_weight_updates']} | {result['weight_min']}/{result['weight_max']}/{result['weight_sum']} | "
            f"{result['status']} |"
        )
    lines.extend(["", "## Confusion Matrix", "", "Python reference, rows=true labels, columns=predicted labels.", ""])
    lines.append("| true\\pred | " + " | ".join(str(i) for i in range(10)) + " |")
    lines.append("|---|" + "|".join(["---:"] * 10) + "|")
    for idx, row in enumerate(py_result.confusion_matrix):
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- RTL uses real `core_group` local weight memory for input-to-output synapses.",
            "- The small RTL shadow weight array is bookkeeping only because `core_group` has no readback port.",
            "- Labels are used only during post-training assignment and evaluation.",
        ]
    )
    (OUT_DIR / "mnist_coregroup_classifier_results.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    train_events = read_events(DATA_DIR / "mnist_classifier_train_1000.mem")
    test_events = read_events(DATA_DIR / "mnist_classifier_test_1000.mem")
    train_labels = read_labels(DATA_DIR / "mnist_classifier_train_1000_labels.txt")
    test_labels = read_labels(DATA_DIR / "mnist_classifier_test_1000_labels.txt")
    py_result = train_assign_test(train_events, train_labels, test_events, test_labels)
    rtl_results = parse_rtl_logs()
    write_outputs(py_result, rtl_results)

    print(f"Python coregroup-like accuracy: {py_result.accuracy:.4f}")
    for subset, result in sorted(rtl_results.items(), key=lambda item: int(item[0])):
        print(
            f"RTL {subset}: accuracy={result['accuracy']:.4f} "
            f"latency={result['avg_latency_cycles']} cycles status={result['status']}"
        )
    print(f"Wrote {OUT_DIR / 'mnist_coregroup_classifier_results.md'}")
    print(f"Wrote {OUT_DIR / 'mnist_coregroup_classifier_results.json'}")
    return 0 if all(result.get("status") == "PASS" for result in rtl_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
