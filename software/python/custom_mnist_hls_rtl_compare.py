#!/usr/bin/env python
"""Compare the one-image MNIST-style RTL spike-file smoke test with Python."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PY_SRC = REPO_ROOT / "software" / "python"
sys.path.insert(0, str(PY_SRC))

from snn_fpga_accelerator.hw_accurate_simulator import (  # noqa: E402
    HWAccurateSNNSimulator,
    LIFNeuronParams,
)


SPIKE_FILE = REPO_ROOT / "hardware" / "hdl" / "tb" / "data" / "mnist_one_image_spikes.mem"
RTL_LOG = REPO_ROOT / "hardware" / "sim_work_custom" / "sim_tb_mnist_hls_rtl_sim.log"


def read_events() -> list[tuple[int, int, int]]:
    events: list[tuple[int, int, int]] = []
    for line in SPIKE_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cycle, neuron_id, weight = (int(part) for part in line.split())
        events.append((cycle, neuron_id, weight))
    return events


def latest_rtl_count() -> int | None:
    if not RTL_LOG.exists():
        return None
    text = RTL_LOG.read_text(errors="ignore")
    matches = re.findall(r"RTL MNIST-style spike count\s*:\s*(\d+)", text)
    return int(matches[-1]) if matches else None


def main() -> int:
    events = read_events()
    params = LIFNeuronParams(threshold=10, leak_rate=0, refractory_period=3)
    sim = HWAccurateSNNSimulator(
        num_groups=2,
        neurons_per_group=16,
        neuron_params=params,
    )

    spike_train: dict[int, list[tuple[int, int, bool]]] = {}
    for cycle, neuron_id, weight in events:
        spike_train.setdefault(cycle, []).append((neuron_id, weight, True))

    sim.reset()
    run_cycles = max((cycle for cycle, _, _ in events), default=0) + 20
    result = sim.run(run_cycles, input_spike_train=spike_train)
    python_count = int(result["per_group_spikes"][0])
    rtl_count = latest_rtl_count()
    matches = rtl_count == python_count

    print("MNIST-style fixed HLS/RTL comparison")
    print("=" * 44)
    print(f"spike file              : {SPIKE_FILE}")
    print(f"input events            : {len(events)}")
    print(f"Python group0 spikes    : {python_count}")
    print(f"RTL group0 spikes       : {rtl_count if rtl_count is not None else 'no xsim log'}")
    print(f"Python matches RTL      : {'YES' if matches else 'NO'}")
    print("=" * 44)
    print("PASS" if matches else "FAIL")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
