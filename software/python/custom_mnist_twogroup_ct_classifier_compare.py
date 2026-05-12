"""Reference/report helper for the two-group CT MNIST classifier.

The Python model mirrors the current RTL trend rather than every pipeline
cycle: group-0 input spikes fan out through a CT-like 64x10 table, group-1
LIF-like output neurons spike naturally, window-level WTA chooses one winner,
and learning updates inter-group weights with saturation, homeostasis, LTD,
and a soft incoming-weight cap.
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
GROUP1_THRESHOLD = 16
W_MIN = 0
W_MAX = 15
A_PLUS = 2
A_MINUS = 1
NORMALIZE_SUM_MAX = 700
REFRACTORY = 8
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
    assignment_winner_counts: List[int]
    dead_outputs: int
    dominant_wins: int
    dominant_ratio: float
    average_input_spikes_per_image: float
    average_group0_spikes_per_image: float
    average_group1_spikes_per_image: float
    average_latency_cycles: float
    average_latency_us_100mhz: float
    throughput_images_per_second: float
    learned_ct_updates: int
    weight_summary: Dict[str, float]


def initial_weight(out_idx: int, in_idx: int) -> int:
    row = in_idx >> 3
    col = in_idx & 7
    center_r = (out_idx * 5 + 1) % 8
    center_c = (out_idx * 3 + 2) % 8
    dist = abs(row - center_r) + abs(col - center_c)
    pattern = (out_idx * 11 + in_idx * 7) % 3
    weight = 6 + pattern
    if dist < 7:
        weight += 7 - dist
    return min(W_MAX, weight)


def make_ct_weights() -> List[List[int]]:
    return [[initial_weight(out_idx, in_idx) for in_idx in range(INPUT_NEURONS)] for out_idx in range(OUTPUT_NEURONS)]


def fanout_order(src_idx: int) -> List[int]:
    by_fanout = [0] * OUTPUT_NEURONS
    for out_idx in range(OUTPUT_NEURONS):
        fanout_idx = (src_idx + out_idx) % OUTPUT_NEURONS
        by_fanout[fanout_idx] = out_idx
    return by_fanout


def run_image(image: data_ref.ImageEvents, weights: Sequence[Sequence[int]]) -> Tuple[int, bool, int, int, int, List[int]]:
    membrane = [0] * OUTPUT_NEURONS
    refractory = [0] * OUTPUT_NEURONS
    output_spikes = [0] * OUTPUT_NEURONS
    group0_spikes = 0
    group1_spikes = 0
    first_latency = 0
    cycle = 0

    for in_idx, active_weight in enumerate(image.weights):
        if active_weight <= 0:
            continue
        group0_spikes += 1
        for out_idx in fanout_order(in_idx):
            cycle += 1
            for ref_idx in range(OUTPUT_NEURONS):
                if refractory[ref_idx] > 0:
                    refractory[ref_idx] -= 1
            membrane[out_idx] += int(weights[out_idx][in_idx])
            if refractory[out_idx] == 0 and membrane[out_idx] >= GROUP1_THRESHOLD:
                output_spikes[out_idx] += 1
                group1_spikes += 1
                membrane[out_idx] = 0
                refractory[out_idx] = REFRACTORY
                if first_latency == 0:
                    first_latency = cycle

    winner, have_winner = choose_inference_winner(output_spikes)
    return winner, have_winner, first_latency if first_latency else max(1, cycle), group0_spikes, group1_spikes, output_spikes


def choose_training_winner(output_spikes: Sequence[int], train_counts: Sequence[int]) -> Tuple[int, bool]:
    candidates = [idx for idx, count in enumerate(output_spikes) if count > 0]
    if not candidates:
        return 0, False
    return max(candidates, key=lambda idx: (-train_counts[idx], output_spikes[idx], -idx)), True


def choose_inference_winner(output_spikes: Sequence[int]) -> Tuple[int, bool]:
    candidates = [idx for idx, count in enumerate(output_spikes) if count > 0]
    if not candidates:
        return 0, False
    return max(candidates, key=lambda idx: (output_spikes[idx], -idx)), True


def apply_stdp(image: data_ref.ImageEvents, weights: List[List[int]], winner: int, output_spikes: Sequence[int], train_counts: Sequence[int]) -> int:
    updates = 0
    floor = min(train_counts)
    row_sums = [sum(row) for row in weights]
    winner_overused = train_counts[winner] > floor

    for in_idx, trace in enumerate(image.weights):
        if trace <= 0:
            continue
        for out_idx in range(OUTPUT_NEURONS):
            old = weights[out_idx][in_idx]
            new = old
            if out_idx == winner:
                if winner_overused:
                    new = max(W_MIN, old - A_MINUS)
                elif row_sums[out_idx] < NORMALIZE_SUM_MAX:
                    new = min(W_MAX, old + A_PLUS + (trace >> 3))
            elif output_spikes[out_idx] > 0 and train_counts[out_idx] > floor:
                new = max(W_MIN, old - A_MINUS)

            if new != old:
                updates += 1
                row_sums[out_idx] += new - old
                weights[out_idx][in_idx] = new
    return updates


def train_assign_test(
    train_events: List[data_ref.ImageEvents],
    train_labels: Sequence[int],
    test_events: List[data_ref.ImageEvents],
    test_labels: Sequence[int],
) -> TwoGroupRun:
    weights = make_ct_weights()
    updates = 0
    train_counts = [0 for _ in range(OUTPUT_NEURONS)]

    for image in train_events:
        _first, have_winner, _latency, _g0, _g1, output_spikes = run_image(image, weights)
        winner, have_winner = choose_training_winner(output_spikes, train_counts)
        if have_winner:
            updates += apply_stdp(image, weights, winner, output_spikes, train_counts)
            train_counts[winner] += 1

    assign_counts = [[0 for _ in range(10)] for _ in range(OUTPUT_NEURONS)]
    assignment_winner_counts = [0 for _ in range(OUTPUT_NEURONS)]
    for image, label in zip(train_events, train_labels):
        _first, _have, _latency, _g0, _g1, output_spikes = run_image(image, weights)
        winner, have_winner = choose_inference_winner(output_spikes)
        if have_winner:
            assign_counts[winner][label] += 1
            assignment_winner_counts[winner] += 1

    assigned_labels = [max(range(10), key=lambda digit: assign_counts[out_idx][digit]) for out_idx in range(OUTPUT_NEURONS)]

    confusion = [[0 for _ in range(10)] for _ in range(10)]
    correct = 0
    latency_sum = 0
    group0_sum = 0
    group1_sum = 0
    for image, label in zip(test_events, test_labels):
        _first, _have, latency, g0_spikes, g1_spikes, output_spikes = run_image(image, weights)
        winner, have_winner = choose_inference_winner(output_spikes)
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
    dominant_wins = max(assignment_winner_counts) if assignment_winner_counts else 0
    dead_outputs = sum(1 for count in assignment_winner_counts if count == 0)
    return TwoGroupRun(
        train_images=len(train_events),
        test_images=len(test_events),
        accuracy=correct / max(1, len(test_events)),
        confusion_matrix=confusion,
        per_class_accuracy=per_class,
        assigned_labels=assigned_labels,
        train_winner_counts=train_counts,
        assignment_winner_counts=assignment_winner_counts,
        dead_outputs=dead_outputs,
        dominant_wins=dominant_wins,
        dominant_ratio=dominant_wins / max(1, len(train_events)),
        average_input_spikes_per_image=sum(img.input_spikes for img in test_events) / max(1, len(test_events)),
        average_group0_spikes_per_image=group0_sum / max(1, len(test_events)),
        average_group1_spikes_per_image=group1_sum / max(1, len(test_events)),
        average_latency_cycles=avg_latency,
        average_latency_us_100mhz=avg_latency / (CLK_HZ / 1_000_000),
        throughput_images_per_second=CLK_HZ / max(1.0, avg_latency),
        learned_ct_updates=updates,
        weight_summary={"min": min(flat), "max": max(flat), "mean": sum(flat) / len(flat), "sum": sum(flat)},
    )


def parse_count_line(text: str, label: str) -> List[int]:
    match = re.search(rf"{label}:\s+([0-9 ]+)", text)
    return [int(item) for item in match.group(1).split()] if match else []


def parse_rtl_logs() -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for log in SIM_DIR.glob("sim_tb_custom_rtl_mnist_twogroup_ct_classifier_*.log"):
        text = log.read_text(encoding="ascii", errors="ignore")
        match = re.search(r"MNIST_TWOGROUP_CT_RTL_SUMMARY\s+(.+)", text)
        if not match:
            continue
        fields: Dict[str, object] = {}
        for token in match.group(1).split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value if key == "subset" else int(value)

        subset = str(fields["subset"])
        latency = int(fields["avg_latency_cycles"])
        results[subset] = {
            "log": str(log),
            "train_images": fields["train_images"],
            "test_images": fields["test_images"],
            "accuracy": fields["accuracy_permille"] / 1000.0,
            "avg_latency_cycles": latency,
            "avg_latency_us_100mhz": latency / (CLK_HZ / 1_000_000),
            "avg_output_spikes_per_image": fields["avg_output_spikes_x1000"] / 1000.0,
            "avg_input_spikes_per_image": fields["avg_input_spikes_x1000"] / 1000.0,
            "avg_group0_spikes_per_image": fields["avg_group0_spikes_x1000"] / 1000.0,
            "avg_group1_spikes_per_image": fields["avg_group1_spikes_x1000"] / 1000.0,
            "throughput_images_per_second": fields["throughput_images_per_sec"],
            "ct_entries": fields["ct_entries"],
            "weight_min": fields["weight_min"],
            "weight_max": fields["weight_max"],
            "weight_sum": fields["weight_sum"],
            "ct_changed_weights": fields["ct_changed_weights"],
            "ct_learned_updates": fields["ct_learned_updates"],
            "ct_init_writes": fields["ct_init_writes"],
            "router_routed_spikes": fields["router_routed_spikes"],
            "router_observed_spikes": fields["router_observed_spikes"],
            "inter_group_routed_spikes": fields["inter_group_routed_spikes"],
            "direct_learning_writes": fields["direct_learning_writes"],
            "direct_ct_writes": fields["direct_ct_writes"],
            "dead_outputs": fields.get("dead_outputs", 0),
            "dominant_output": fields.get("dominant_output", 0),
            "dominant_wins": fields.get("dominant_wins", 0),
            "dominant_ratio": fields.get("dominant_ratio_permille", 0) / 1000.0,
            "train_dead_outputs": fields.get("train_dead_outputs", 0),
            "train_dominant_wins": fields.get("train_dominant_wins", 0),
            "assignment_winner_counts": parse_count_line(text, "TWOGROUP_CT_ASSIGN_WIN_COUNTS"),
            "assigned_labels": parse_count_line(text, "TWOGROUP_CT_ASSIGNED_LABELS"),
            "train_winner_counts": parse_count_line(text, "TWOGROUP_CT_TRAIN_WIN_COUNTS"),
            "group1_input_counts": parse_count_line(text, "TWOGROUP_CT_GROUP1_INPUT_COUNTS"),
            "group1_output_counts": parse_count_line(text, "TWOGROUP_CT_GROUP1_OUTPUT_COUNTS"),
            "sim_cycles": fields["sim_cycles"],
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
            "wta": "window-level natural-spike WTA",
            "homeostasis": "training winner selection prefers least-used natural spiking outputs",
            "ltd": "overused winner/non-winner depression",
            "normalization": f"incoming row cap {NORMALIZE_SUM_MAX}",
            "learn_weight_is_inter": True,
            "direct_learning_writes": False,
        },
        "python_reference": asdict(py_result),
        "rtl_results": rtl_results,
        "python_comparison_status": {subset: comparison_status(result) for subset, result in rtl_results.items()},
    }
    (OUT_DIR / "mnist_twogroup_ct_classifier_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

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
        "- WTA: window-level winner from natural group-1 spikes.",
        "- Homeostasis: training winner selection favors least-used spiking outputs.",
        "- LTD/normalization: overused active outputs can be depressed; incoming row sum is capped.",
        "",
        "## Python Reference",
        "",
        f"- Train images: {py_result.train_images}",
        f"- Test images: {py_result.test_images}",
        f"- Accuracy: {py_result.accuracy:.4f}",
        f"- Assignment winner counts: {py_result.assignment_winner_counts}",
        f"- Dead outputs: {py_result.dead_outputs}",
        f"- Dominant ratio: {py_result.dominant_ratio:.3f}",
        f"- Average input spikes/image: {py_result.average_input_spikes_per_image:.2f}",
        f"- Average group0 spikes/image: {py_result.average_group0_spikes_per_image:.2f}",
        f"- Average group1 spikes/image: {py_result.average_group1_spikes_per_image:.2f}",
        f"- Learned CT updates: {py_result.learned_ct_updates}",
        "",
        "## RTL Representative Runs",
        "",
        "| subset | train | test | accuracy | latency cycles | latency us | throughput img/s | input spikes/img | group0 spikes/img | group1 spikes/img | inter routed | CT updates | dead outputs | dominant ratio | direct writes | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for subset in sorted(rtl_results, key=lambda s: int(s)):
        result = rtl_results[subset]
        direct_writes = int(result["direct_learning_writes"]) + int(result["direct_ct_writes"])
        lines.append(
            f"| {subset} | {result['train_images']} | {result['test_images']} | {result['accuracy']:.4f} | "
            f"{result['avg_latency_cycles']} | {result['avg_latency_us_100mhz']:.3f} | {result['throughput_images_per_second']} | "
            f"{result['avg_input_spikes_per_image']:.2f} | {result['avg_group0_spikes_per_image']:.2f} | "
            f"{result['avg_group1_spikes_per_image']:.2f} | {result['inter_group_routed_spikes']} | "
            f"{result['ct_learned_updates']} | {result['dead_outputs']} | {result['dominant_ratio']:.3f} | "
            f"{direct_writes} | {comparison_status(result)} |"
        )
        lines.append("")
        lines.append(f"RTL `{subset}` assignment winner counts: {result.get('assignment_winner_counts', [])}")
        lines.append(f"RTL `{subset}` assigned labels: {result.get('assigned_labels', [])}")
        lines.append(f"RTL `{subset}` training winner counts: {result.get('train_winner_counts', [])}")
        lines.append("")

    lines.extend(
        [
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
            "- Labels are used only after unsupervised training for output-neuron assignment.",
            "- Accuracy is still low; this pass mainly improves output-neuron usage and reduces one-neuron collapse.",
            "- The next tuning target is better prototype formation before scaling to four or more groups.",
        ]
    )
    (OUT_DIR / "mnist_twogroup_ct_classifier_results.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    train_events = data_ref.read_events(DATA_DIR / "mnist_classifier_train_1000.mem")
    test_events = data_ref.read_events(DATA_DIR / "mnist_classifier_test_1000.mem")
    train_labels = data_ref.read_labels(DATA_DIR / "mnist_classifier_train_1000_labels.txt")
    test_labels = data_ref.read_labels(DATA_DIR / "mnist_classifier_test_1000_labels.txt")
    py_result = train_assign_test(train_events, train_labels, test_events, test_labels)
    rtl_results = parse_rtl_logs()
    write_outputs(py_result, rtl_results)

    print(f"Python two-group CT reference accuracy: {py_result.accuracy:.4f}")
    print(f"Python assignment winner counts: {py_result.assignment_winner_counts}")
    for subset, result in sorted(rtl_results.items(), key=lambda item: int(item[0])):
        print(
            f"RTL CT {subset}: accuracy={result['accuracy']:.4f} "
            f"latency={result['avg_latency_cycles']} cycles "
            f"dead_outputs={result['dead_outputs']} "
            f"dominant_ratio={result['dominant_ratio']:.3f} "
            f"ct_updates={result['ct_learned_updates']} "
            f"status={comparison_status(result)}"
        )
    print(f"Wrote {OUT_DIR / 'mnist_twogroup_ct_classifier_results.md'}")
    print(f"Wrote {OUT_DIR / 'mnist_twogroup_ct_classifier_results.json'}")
    return 0 if all(comparison_status(result) == "PASS" for result in rtl_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
