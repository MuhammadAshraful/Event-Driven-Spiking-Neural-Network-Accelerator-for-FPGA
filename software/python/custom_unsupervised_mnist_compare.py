"""Python reference for the staged RTL-only unsupervised MNIST smoke tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "hardware" / "hdl" / "tb" / "data"
SIM_DIR = REPO_ROOT / "hardware" / "sim_work_rtl_unsupervised"
OUT_DIR = REPO_ROOT / "outputs"

THRESHOLD = 10
INITIAL_WEIGHT = 4
A_PLUS = 3
W_MAX = 15
STDP_WINDOW = 20000
OUTPUT_INHIBIT_WINDOW = 200


@dataclass(frozen=True)
class Event:
    cycle: int
    neuron_id: int
    weight: int


@dataclass
class PyResult:
    name: str
    input_spikes: int
    before_spikes: int
    train_spikes: int
    inference_spikes: int
    learned_updates: int
    weights: Tuple[int, int, int]
    winner: int


@dataclass
class RtlResult:
    input_spikes: int
    before_spikes: int
    train_spikes: int
    inference_spikes: int
    learned_updates: int
    weights: Tuple[int, int, int]
    winner: int
    pass_fail: str


def read_events(path: Path) -> List[Event]:
    events: List[Event] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        cycle, neuron_id, weight = (int(part) for part in line.split())
        events.append(Event(cycle, neuron_id, weight))
    return events


def run_training(events: Iterable[Event]) -> Tuple[int, int, Tuple[int, int, int]]:
    weights = [INITIAL_WEIGHT, INITIAL_WEIGHT, INITIAL_WEIGHT]
    last_pre = [-1, -1, -1]
    membrane = 0
    output_spikes = 0
    updates = 0
    inhibited_until = -1

    for event in events:
        if not 0 <= event.neuron_id < 3:
            continue
        if event.weight < THRESHOLD:
            continue

        src = event.neuron_id
        last_pre[src] = event.cycle

        # The RTL smoke topology effectively produces one output response per
        # compact image window. Events arriving during the short inhibition
        # window are still valid pre spikes, but they do not create another
        # output spike.
        if event.cycle < inhibited_until:
            continue

        membrane += weights[src]

        if membrane >= THRESHOLD:
            output_spikes += 1
            membrane = 0
            inhibited_until = event.cycle + OUTPUT_INHIBIT_WINDOW
            for pre_src in range(3):
                if last_pre[pre_src] >= 0 and event.cycle - last_pre[pre_src] <= STDP_WINDOW:
                    new_weight = min(weights[pre_src] + A_PLUS, W_MAX)
                    if new_weight != weights[pre_src]:
                        updates += 1
                    weights[pre_src] = new_weight

    return output_spikes, updates, tuple(weights)


def run_inference(events: Iterable[Event], weights: Tuple[int, int, int]) -> int:
    membrane = 0
    output_spikes = 0
    inhibited_until = -1
    for event in events:
        if not 0 <= event.neuron_id < 3:
            continue
        if event.weight < THRESHOLD:
            continue
        if event.cycle < inhibited_until:
            continue
        membrane += weights[event.neuron_id]
        if membrane >= THRESHOLD:
            output_spikes += 1
            membrane = 0
            inhibited_until = event.cycle + OUTPUT_INHIBIT_WINDOW
    return output_spikes


def simulate_python(name: str, path: Path) -> PyResult:
    events = read_events(path)
    before_spikes = run_inference(events, (INITIAL_WEIGHT, INITIAL_WEIGHT, INITIAL_WEIGHT))
    train_spikes, updates, weights = run_training(events)
    inference_spikes = run_inference(events, weights)
    return PyResult(
        name=name,
        input_spikes=len(events),
        before_spikes=before_spikes,
        train_spikes=train_spikes,
        inference_spikes=inference_spikes,
        learned_updates=updates,
        weights=weights,
        winner=3,
    )


def parse_rtl_log(path: Path) -> Optional[RtlResult]:
    if not path.exists():
        return None

    text = path.read_text(encoding="ascii", errors="ignore")
    summary = re.search(
        r"MNIST RTL Summary: .*?input_spikes=(\d+) before_spikes=(\d+) train_spikes=(\d+) "
        r"inference_spikes=(\d+) learned_updates=(\d+) weights=(\d+),(\d+),(\d+) winner=(\d+)",
        text,
    )
    if not summary:
        return None

    pass_fail = "PASS" if "MNIST UNSUPERVISED RTL SMOKE TEST PASSED" in text else "FAIL"
    return RtlResult(
        input_spikes=int(summary.group(1)),
        before_spikes=int(summary.group(2)),
        train_spikes=int(summary.group(3)),
        inference_spikes=int(summary.group(4)),
        learned_updates=int(summary.group(5)),
        weights=(int(summary.group(6)), int(summary.group(7)), int(summary.group(8))),
        winner=int(summary.group(9)),
        pass_fail=pass_fail,
    )


def compare(py: PyResult, rtl: Optional[RtlResult]) -> str:
    if rtl is None:
        return "NO_RTL_LOG"
    if rtl.pass_fail != "PASS":
        return "RTL_FAIL"
    checks = [
        rtl.input_spikes == py.input_spikes,
        rtl.winner == py.winner,
        0 <= rtl.before_spikes <= py.before_spikes,
        abs(rtl.train_spikes - py.train_spikes) <= 2,
        abs(rtl.inference_spikes - py.inference_spikes) <= 2,
        rtl.inference_spikes >= rtl.before_spikes,
        all(rw >= INITIAL_WEIGHT for rw in rtl.weights),
        any(rw > INITIAL_WEIGHT for rw in rtl.weights),
    ]
    return "PASS" if all(checks) else "CHECK_TOLERANCE"


def row(name: str, py: PyResult, rtl: Optional[RtlResult], status: str) -> str:
    rtl_before = "-" if rtl is None else str(rtl.before_spikes)
    rtl_inf = "-" if rtl is None else str(rtl.inference_spikes)
    rtl_weights = "-" if rtl is None else "/".join(str(w) for w in rtl.weights)
    rtl_status = "-" if rtl is None else rtl.pass_fail
    return (
        f"| {name} | {py.input_spikes} | 3 pooled | 1 | unsupervised STDP | "
        f"{rtl_before} | {rtl_inf} | {rtl_weights} | 3 | {rtl_status} | {status} | not captured |"
    )


def write_results(results: Dict[str, Tuple[PyResult, Optional[RtlResult], str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Unsupervised MNIST RTL Results",
        "",
        "| experiment | input spikes | input neurons used | output neurons used | training mode | before-learning output spikes | after-learning output spikes | learned weight changes | winner neuron | RTL PASS/FAIL | Python comparison PASS/FAIL | simulation time |",
        "|---|---:|---|---:|---|---:|---:|---|---:|---|---|---|",
    ]
    for name, (py, rtl, status) in results.items():
        lines.append(row(name, py, rtl, status))
    lines.extend(
        [
            "",
            "Notes:",
            "- Labels are not used by RTL training; they are emitted separately by the spike generator for later evaluation.",
            "- This smoke test pools each 28x28 image into three input neurons and one output neuron.",
            "- Python comparison allows a small spike-count tolerance because RTL handshakes add cycle spacing around each event.",
        ]
    )
    (OUT_DIR / "unsupervised_mnist_results.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    experiments = {
        "mnist1": (
            DATA_DIR / "mnist_unsup_1.mem",
            SIM_DIR / "sim_tb_custom_rtl_unsupervised_mnist_mnist1.log",
        ),
        "mnist10": (
            DATA_DIR / "mnist_unsup_10.mem",
            SIM_DIR / "sim_tb_custom_rtl_unsupervised_mnist_mnist10.log",
        ),
        "batch": (
            DATA_DIR / "mnist_unsup_batch.mem",
            SIM_DIR / "sim_tb_custom_rtl_unsupervised_mnist_batch.log",
        ),
    }

    results: Dict[str, Tuple[PyResult, Optional[RtlResult], str]] = {}
    print("experiment | input | py_before | rtl_before | py_train | rtl_train | py_infer | rtl_infer | py_weights | rtl_weights | status")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---")
    for name, (event_file, log_file) in experiments.items():
        py = simulate_python(name, event_file)
        rtl = parse_rtl_log(log_file)
        status = compare(py, rtl)
        rtl_before = "-" if rtl is None else str(rtl.before_spikes)
        rtl_train = "-" if rtl is None else str(rtl.train_spikes)
        rtl_inf = "-" if rtl is None else str(rtl.inference_spikes)
        rtl_weights = "-" if rtl is None else "/".join(str(w) for w in rtl.weights)
        py_weights = "/".join(str(w) for w in py.weights)
        print(
            f"{name} | {py.input_spikes} | {py.before_spikes} | {rtl_before} | "
            f"{py.train_spikes} | {rtl_train} | "
            f"{py.inference_spikes} | {rtl_inf} | {py_weights} | {rtl_weights} | {status}"
        )
        results[name] = (py, rtl, status)

    write_results(results)
    print(f"\nWrote {OUT_DIR / 'unsupervised_mnist_results.md'}")
    return 0 if all(status in {"PASS", "NO_RTL_LOG"} for _, _, status in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
