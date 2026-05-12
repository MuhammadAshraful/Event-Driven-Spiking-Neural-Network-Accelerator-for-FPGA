"""Python comparison/report helper for the two-group CT MNIST classifier.

This approximates the new RTL milestone:
MNIST events make group-0 input neurons fire, event_router_ng + CT fanout sends
weighted spikes to group 1, group-1 LIF-like membranes produce output spikes,
WTA chooses a winner, and active input-to-winner CT weights are strengthened.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import custom_mnist_coregroup_classifier_compare as data_ref


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "hardware" / "hdl" / "tb" / "data"
SIM_DIR = REPO_ROOT / "hardware" / "sim_work_rtl_mnist_twogroup_ct_classifier"
OUT_DIR = REPO_ROOT / "outputs"

INPUT_NEURONS = 64
OUTPUT_NEURONS = 10
CT_ENTRIES = INPUT_NEURONS * OUTPUT_NEURONS
GROUP0_THRESHOLD = 8
GROUP1_THRESHOLD = 96
W_MIN = 0
W_MAX = 15
A_PLUS = 2
CLK_HZ = 100_000_000


@dataclass
class TwoGroupRun:
    train_images: int
    test_images: int
    accuracy: float
    confusion_matrix: List[List[int]]
    per_class_accuracy: Dict[str, float]
    assigned_labels: List[int]
    train_winner_counts: List[int]
    average_input_spikes_per_image: float
    average_group0_spikes_per_image: float
    average_group1_spikes_per_image: float
    average_latency_cycles: float
    average_latency_us_100mhz: float
    throughput_images_per_second: float
    learned_ct_updates: int
    weight_summary: Dict[str, float]


def initial_weight(out_idx: int, in_idx: int) -> int:
    pattern = (out_idx * 13 + in_idx * 7 + (in_idx >> 3) * 5 + (in_idx & 7) * 3) % 10
    return 4 + pattern


def make_ct_weights() -> List[List[int]]:
    return [[initial_weight(out_idx, in_idx) for in_idx in range(INPUT_NEURONS)] for out_idx in range(OUTPUT_NEURONS)]


def run_image(image: data_ref.ImageEvents, weights: Sequence[Sequence[int]]) -> Tuple[int, bool, int, int, int]:
    membrane = [0] * OUTPUT_NEURONS
    refractory = [0] * OUTPUT_NEURONS
    first_winner = 0
    have_winner = False
    group0_spikes = 0
    group1_spikes = 0
    latency = 0
    cycle = 0

    for in_idx, active_weight in enumerate(image.weights):
        if active_weight <= 0:
            continue
        group0_spikes += 1
        for out_idx in range(OUTPUT_NEURONS):
            cycle += 1
            if refractory[out_idx] > 0:
                refractory[out_idx] -= 1
            membrane[out_idx] += int(weights[out_idx][in_idx])
            if refractory[out_idx] == 0 and membrane[out_idx] >= GROUP1_THRESHOLD:
                group1_spikes += 1
                membrane[out_idx] = 0
                refractory[out_idx] = 8
                if not have_winner:
                    first_winner = out_idx
                    have_winner = True
                    latency = cycle
    return first_winner, have_winner, latency if have_winner else max(1, cycle), group0_spikes, group1_spikes


def apply_active_stdp(image: data_ref.ImageEvents, weights: List[List[int]], winner: int) -> int:
    updates = 0
    for in_idx, trace in enumerate(image.weights):
        if trace <= 0:
            continue
        old = weights[winner][in_idx]
        new = min(W_MAX, old + A_PLUS + (trace >> 3))
        if new != old:
            updates += 1
        weights[winner][in_idx] = new
    return updates


def train_assign_test(
    train_events: List[data_ref.ImageEvents],
    train_labels: Sequence[int],
    test_events: List[data_ref.ImageEvents],
    test_labels: Sequence[int],
) -> TwoGroupRun:
    weights = make_ct_weights()
    updates = 0

    for image in train_events:
        winner, have_winner, _latency, _g0, _g1 = run_image(image, weights)
        if have_winner:
            updates += apply_active_stdp(image, weights, winner)

    assign_counts = [[0 for _ in range(10)] for _ in range(OUTPUT_NEURONS)]
    train_winner_counts = [0 for _ in range(OUTPUT_NEURONS)]
    for image, label in zip(train_events, train_labels):
        winner, have_winner, _latency, _g0, _g1 = run_image(image, weights)
        if have_winner:
            assign_counts[winner][label] += 1
            train_winner_counts[winner] += 1

    assigned_labels: List[int] = []
    for out_idx in range(OUTPUT_NEURONS):
        assigned_labels.append(max(range(10), key=lambda digit: assign_counts[out_idx][digit]))

    confusion = [[0 for _ in range(10)] for _ in range(10)]
    correct = 0
    latency_sum = 0
    group0_sum = 0
    group1_sum = 0
    for image, label in zip(test_events, test_labels):
        winner, have_winner, latency, g0_spikes, g1_spikes = run_image(image, weights)
        pred = assigned_labels[winner] if have_winner else 0
        confusion[label][pred] += 1
        correct += int(pred == label)
        latency_sum += latency
        group0_sum += g0_spikes
        group1_sum += g1_spikes

    per_class: Dict[str, float] = {}
    for digit in range(10):
        total = sum(confusion[digit])
        per_class[str(digit)] = confusion[digit][digit] / total if total else 0.0

    flat = [w for row in weights for w in row]
    avg_latency = latency_sum / max(1, len(test_events))
    return TwoGroupRun(
        train_images=len(train_events),
        test_images=len(test_events),
        accuracy=correct / max(1, len(test_events)),
        confusion_matrix=confusion,
        per_class_accuracy=per_class,
        assigned_labels=assigned_labels,
        train_winner_counts=train_winner_counts,
        average_input_spikes_per_image=sum(img.input_spikes for img in test_events) / max(1, len(test_events)),
        average_group0_spikes_per_image=group0_sum / max(1, len(test_events)),
        average_group1_spikes_per_image=group1_sum / max(1, len(test_events)),
        average_latency_cycles=avg_latency,
        average_latency_us_100mhz=avg_latency / (CLK_HZ / 1_000_000),
        throughput_images_per_second=CLK_HZ / max(1.0, avg_latency),
        learned_ct_updates=updates,
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
        r"MNIST_TWOGROUP_CT_RTL_SUMMARY subset=(\S+) train_images=(\d+) test_images=(\d+) "
        r"accuracy_permille=(\d+) avg_latency_cycles=(\d+) avg_output_spikes_x1000=(\d+) "
        r"avg_input_spikes_x1000=(\d+) avg_group0_spikes_x1000=(\d+) avg_group1_spikes_x1000=(\d+) "
        r"throughput_images_per_sec=(\d+) ct_entries=(\d+) weight_min=(\d+) weight_max=(\d+) "
        r"weight_sum=(\d+) ct_changed_weights=(\d+) ct_learned_updates=(\d+) ct_init_writes=(\d+) "
        r"router_routed_spikes=(\d+) router_observed_spikes=(\d+) inter_group_routed_spikes=(\d+) "
        r"direct_learning_writes=(\d+) direct_ct_writes=(\d+) sim_cycles=(\d+)"
    )
    for log in SIM_DIR.glob("sim_tb_custom_rtl_mnist_twogroup_ct_classifier_*.log"):
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
            "avg_group0_spikes_per_image": int(match.group(8)) / 1000.0,
            "avg_group1_spikes_per_image": int(match.group(9)) / 1000.0,
            "throughput_images_per_second": int(match.group(10)),
            "ct_entries": int(match.group(11)),
            "weight_min": int(match.group(12)),
            "weight_max": int(match.group(13)),
            "weight_sum": int(match.group(14)),
            "ct_changed_weights": int(match.group(15)),
            "ct_learned_updates": int(match.group(16)),
            "ct_init_writes": int(match.group(17)),
            "router_routed_spikes": int(match.group(18)),
            "router_observed_spikes": int(match.group(19)),
            "inter_group_routed_spikes": int(match.group(20)),
            "direct_learning_writes": int(match.group(21)),
            "direct_ct_writes": int(match.group(22)),
            "sim_cycles": int(match.group(23)),
            "status": "PASS" if "MNIST TWOGROUP CT CLASSIFIER RTL TEST PASSED" in text else "FAIL",
        }
    return results


def comparison_status(result: Dict[str, object]) -> str:
    checks = [
        result.get("status") == "PASS",
        result.get("ct_entries") == CT_ENTRIES,
        result.get("ct_init_writes") == CT_ENTRIES,
        result.get("inter_group_routed_spikes", 0) > 0,
        result.get("router_observed_spikes", 0) > 0,
        result.get("ct_learned_updates", 0) > 0,
        result.get("direct_learning_writes") == 0,
        result.get("direct_ct_writes") == 0,
    ]
    return "PASS" if all(checks) else "FAIL"


def write_outputs(py_result: TwoGroupRun, rtl_results: Dict[str, Dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": {
            "active_core_groups": 2,
            "ct_used": True,
            "ct_entries": CT_ENTRIES,
            "input_neurons": INPUT_NEURONS,
            "output_neurons": OUTPUT_NEURONS,
            "learn_weight_is_inter": True,
            "direct_learning_writes": False,
        },
        "python_reference": asdict(py_result),
        "rtl_results": rtl_results,
        "python_comparison_status": {
            subset: comparison_status(result) for subset, result in rtl_results.items()
        },
    }
    (OUT_DIR / "mnist_twogroup_ct_classifier_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )

    lines = [
        "# MNIST Two-Group CT Classifier Results",
        "",
        "## Architecture",
        "",
        "- Active core groups: 2",
        "- Connectivity table: used for group 0 -> group 1 fanout",
        "- CT entries used: 640",
        "- Learned updates use `event_router_ng.learn_weight_*` with `learn_weight_is_inter=1`.",
        "- Direct CT/core_group learning writes are avoided.",
        "",
        "## Python Reference",
        "",
        f"- Train images: {py_result.train_images}",
        f"- Test images: {py_result.test_images}",
        f"- Accuracy: {py_result.accuracy:.4f}",
        f"- Average input spikes/image: {py_result.average_input_spikes_per_image:.2f}",
        f"- Average group0 spikes/image: {py_result.average_group0_spikes_per_image:.2f}",
        f"- Average group1 spikes/image: {py_result.average_group1_spikes_per_image:.2f}",
        f"- Learned CT updates: {py_result.learned_ct_updates}",
        "",
        "## RTL Representative Runs",
        "",
        "| subset | train | test | accuracy | latency cycles | latency us | throughput img/s | input spikes/img | group0 spikes/img | group1 spikes/img | inter routed | CT updates | direct writes | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for subset in sorted(rtl_results, key=lambda s: int(s)):
        result = rtl_results[subset]
        direct_writes = int(result["direct_learning_writes"]) + int(result["direct_ct_writes"])
        lines.append(
            f"| {subset} | {result['train_images']} | {result['test_images']} | "
            f"{result['accuracy']:.4f} | {result['avg_latency_cycles']} | "
            f"{result['avg_latency_us_100mhz']:.3f} | {result['throughput_images_per_second']} | "
            f"{result['avg_input_spikes_per_image']:.2f} | {result['avg_group0_spikes_per_image']:.2f} | "
            f"{result['avg_group1_spikes_per_image']:.2f} | {result['inter_group_routed_spikes']} | "
            f"{result['ct_learned_updates']} | {direct_writes} | {comparison_status(result)} |"
        )

    lines.extend(
        [
            "",
            "## Python Confusion Matrix",
            "",
            "Rows are true labels, columns are predicted labels.",
            "",
            "| true\\pred | " + " | ".join(str(i) for i in range(10)) + " |",
            "|---|" + "|".join(["---:"] * 10) + "|",
        ]
    )
    for idx, row in enumerate(py_result.confusion_matrix):
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Labels are used only after unsupervised training.",
            "- Accuracy is still low because this milestone prioritizes architecture fidelity over tuning.",
            "- The next step is scaling the same CT/router pattern to more core groups.",
        ]
    )
    (OUT_DIR / "mnist_twogroup_ct_classifier_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def main() -> int:
    train_events = data_ref.read_events(DATA_DIR / "mnist_classifier_train_1000.mem")
    test_events = data_ref.read_events(DATA_DIR / "mnist_classifier_test_1000.mem")
    train_labels = data_ref.read_labels(DATA_DIR / "mnist_classifier_train_1000_labels.txt")
    test_labels = data_ref.read_labels(DATA_DIR / "mnist_classifier_test_1000_labels.txt")
    py_result = train_assign_test(train_events, train_labels, test_events, test_labels)
    rtl_results = parse_rtl_logs()
    write_outputs(py_result, rtl_results)

    print(f"Python two-group CT reference accuracy: {py_result.accuracy:.4f}")
    for subset, result in sorted(rtl_results.items(), key=lambda item: int(item[0])):
        print(
            f"RTL CT {subset}: accuracy={result['accuracy']:.4f} "
            f"latency={result['avg_latency_cycles']} cycles "
            f"ct_updates={result['ct_learned_updates']} "
            f"status={comparison_status(result)}"
        )
    print(f"Wrote {OUT_DIR / 'mnist_twogroup_ct_classifier_results.md'}")
    print(f"Wrote {OUT_DIR / 'mnist_twogroup_ct_classifier_results.json'}")
    return 0 if all(comparison_status(result) == "PASS" for result in rtl_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
