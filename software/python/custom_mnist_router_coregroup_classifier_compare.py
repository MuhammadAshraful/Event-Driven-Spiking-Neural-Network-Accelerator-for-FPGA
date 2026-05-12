"""Comparison/report helper for the router + core_group MNIST classifier.

The RTL classifier in this path is intentionally still small: one active
core_group, with event_router_ng handling external spike delivery and learned
intra-group weight writes.  This script reuses the existing core_group-like
Python reference for the algorithmic baseline, then parses the router RTL logs
and writes router-specific result files.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import custom_mnist_coregroup_classifier_compare as core_ref


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = REPO_ROOT / "hardware" / "sim_work_rtl_mnist_router_coregroup_classifier"
OUT_DIR = REPO_ROOT / "outputs"
CLK_HZ = 100_000_000


def parse_rtl_logs() -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    pattern = re.compile(
        r"MNIST_ROUTER_COREGROUP_RTL_SUMMARY subset=(\S+) train_images=(\d+) test_images=(\d+) "
        r"accuracy_permille=(\d+) avg_latency_cycles=(\d+) avg_output_spikes_x1000=(\d+) "
        r"avg_input_spikes_x1000=(\d+) throughput_images_per_sec=(\d+) "
        r"weight_min=(\d+) weight_max=(\d+) weight_sum=(\d+) total_updates=(\d+) "
        r"router_weight_writes=(\d+) router_routed_spikes=(\d+) router_observed_spikes=(\d+) "
        r"direct_learning_writes=(\d+) sim_cycles=(\d+)"
    )
    for log in SIM_DIR.glob("sim_tb_custom_rtl_mnist_router_coregroup_classifier_*.log"):
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
            "router_learning_weight_writes": int(match.group(13)),
            "router_routed_spikes": int(match.group(14)),
            "router_observed_spikes": int(match.group(15)),
            "direct_learning_writes": int(match.group(16)),
            "sim_cycles": int(match.group(17)),
            "status": "PASS" if "MNIST ROUTER COREGROUP CLASSIFIER RTL TEST PASSED" in text else "FAIL",
        }
    return results


def comparison_status(result: Dict[str, object]) -> str:
    checks = [
        result.get("status") == "PASS",
        result.get("train_images") == result.get("test_images"),
        result.get("router_routed_spikes", 0) > 0,
        result.get("router_observed_spikes", 0) > 0,
        result.get("router_learning_weight_writes", 0) > 0,
        result.get("direct_learning_writes") == 0,
        result.get("learned_weight_updates", 0) > 0,
    ]
    return "PASS" if all(checks) else "FAIL"


def write_outputs(py_result: core_ref.ClassifierRun, rtl_results: Dict[str, Dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": {
            "input_neurons": core_ref.INPUT_NEURONS,
            "output_neurons": core_ref.OUTPUT_NEURONS,
            "router_path_used": True,
            "learn_weight_path_used": True,
            "direct_core_group_learning_writes": False,
            "active_core_groups": 1,
            "event_router_num_groups": 2,
        },
        "python_coregroup_like_reference": asdict(py_result),
        "rtl_results": rtl_results,
        "python_comparison_status": {
            subset: comparison_status(result) for subset, result in rtl_results.items()
        },
    }
    (OUT_DIR / "mnist_router_coregroup_classifier_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )

    lines = [
        "# MNIST Router + Coregroup Classifier Results",
        "",
        "## Architecture",
        "",
        "- Active RTL core groups: 1",
        "- event_router_ng instances: 1",
        "- event_router_ng NUM_GROUPS: 2, with only group 0 active",
        "- Input neurons: 64, local IDs 0..63",
        "- Output neurons: 10, local IDs 64..73",
        "- Input spikes enter through event_router_ng external-spike routing.",
        "- Learned weights are issued through event_router_ng.learn_weight_*.",
        "- Direct core_group learning writes are not used in this path.",
        "",
        "## Python Reference",
        "",
        f"- Train images: {py_result.train_images}",
        f"- Test images: {py_result.test_images}",
        f"- Accuracy: {py_result.accuracy:.4f}",
        f"- Average input spikes/image: {py_result.average_input_spikes_per_image:.2f}",
        f"- Average output spikes/image: {py_result.average_output_spikes_per_image:.2f}",
        f"- Learned weight updates: {py_result.learned_weight_updates}",
        "",
        "## RTL Representative Runs",
        "",
        "| subset | train | test | accuracy | latency cycles | latency us | throughput img/s | input spikes/img | output spikes/img | changed weights | router writes | routed spikes | observed spikes | direct writes | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for subset in sorted(rtl_results, key=lambda s: int(s)):
        result = rtl_results[subset]
        lines.append(
            f"| {subset} | {result['train_images']} | {result['test_images']} | "
            f"{result['accuracy']:.4f} | {result['avg_latency_cycles']} | "
            f"{result['avg_latency_us_100mhz']:.3f} | {result['throughput_images_per_second']} | "
            f"{result['avg_input_spikes_per_image']:.2f} | {result['avg_output_spikes_per_image']:.2f} | "
            f"{result['learned_weight_updates']} | {result['router_learning_weight_writes']} | "
            f"{result['router_routed_spikes']} | {result['router_observed_spikes']} | "
            f"{result['direct_learning_writes']} | {comparison_status(result)} |"
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
            "- Labels are used only after unsupervised training for assignment/evaluation.",
            "- The RTL shadow weight array is for logging/update bookkeeping only; spike processing uses core_group weight memory.",
            "- synaptic_connectivity_table is intentionally not in this milestone yet.",
        ]
    )
    (OUT_DIR / "mnist_router_coregroup_classifier_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def main() -> int:
    train_events = core_ref.read_events(core_ref.DATA_DIR / "mnist_classifier_train_1000.mem")
    test_events = core_ref.read_events(core_ref.DATA_DIR / "mnist_classifier_test_1000.mem")
    train_labels = core_ref.read_labels(core_ref.DATA_DIR / "mnist_classifier_train_1000_labels.txt")
    test_labels = core_ref.read_labels(core_ref.DATA_DIR / "mnist_classifier_test_1000_labels.txt")
    py_result = core_ref.train_assign_test(train_events, train_labels, test_events, test_labels)
    rtl_results = parse_rtl_logs()
    write_outputs(py_result, rtl_results)

    print(f"Python coregroup-like reference accuracy: {py_result.accuracy:.4f}")
    for subset, result in sorted(rtl_results.items(), key=lambda item: int(item[0])):
        print(
            f"RTL router {subset}: accuracy={result['accuracy']:.4f} "
            f"latency={result['avg_latency_cycles']} cycles "
            f"router_writes={result['router_learning_weight_writes']} "
            f"status={comparison_status(result)}"
        )
    print(f"Wrote {OUT_DIR / 'mnist_router_coregroup_classifier_results.md'}")
    print(f"Wrote {OUT_DIR / 'mnist_router_coregroup_classifier_results.json'}")
    return 0 if all(comparison_status(result) == "PASS" for result in rtl_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
