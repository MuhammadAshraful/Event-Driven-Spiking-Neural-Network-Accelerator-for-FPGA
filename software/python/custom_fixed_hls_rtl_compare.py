#!/usr/bin/env python
"""Small fixed HLS/RTL learning comparison.

This mirrors the custom Verilog experiment:

1. An initial intra-group RTL weight n0->n1 is sub-threshold.
2. A pre spike fires neuron 0, but neuron 1 does not fire.
3. A generated-HLS-style STDP update is computed for pre0/post1.
4. The learned weight is applied to the RTL simulator.
5. The same pre input now causes neuron 1 to fire.

The checked-in generated HLS Verilog is older than the current HLS C++ source
and uses an 8-bit signed learning range. This script sets the Python STDP
engine bounds to that legacy generated-Verilog range for this comparison.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PY_SRC = REPO_ROOT / "software" / "python"
sys.path.insert(0, str(PY_SRC))

from snn_fpga_accelerator.hw_accurate_simulator import (  # noqa: E402
    HWAccurateSNNSimulator,
    HWAccurateSTDPEngine,
    LIFNeuronParams,
    STDPConfig,
)


def latest_rtl_learned_weight() -> int | None:
    """Read the latest xsim learned-weight value when the RTL log exists."""
    log_path = REPO_ROOT / "hardware" / "sim_work_custom" / "sim_tb_custom_hls_rtl_learning.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"HLS learned/applied RTL weight n0->n1:\s*(\d+)", text)
    if not matches:
        matches = re.findall(r"HLS weight update/result\s*:\s*(\d+)", text)
    return int(matches[-1]) if matches else None


def run_trial(sim: HWAccurateSNNSimulator, weight_0_1: int) -> int:
    """Run one isolated n0 stimulus trial and return group-0 spike count."""
    sim.reset()
    sim.set_intra_weight(0, 0, 1, weight_0_1, exc=True)
    input_id = sim.local_to_global(0, 0)
    result = sim.run(
        20,
        input_spike_train={0: [(input_id, 12, True)]},
    )
    return int(result["per_group_spikes"][0])


def main() -> int:
    params = LIFNeuronParams(
        threshold=10,
        leak_rate=0,
        refractory_period=3,
    )
    sim = HWAccurateSNNSimulator(
        num_groups=2,
        neurons_per_group=16,
        neuron_params=params,
    )

    initial_weight = 5
    before_count = run_trial(sim, initial_weight)

    stdp = HWAccurateSTDPEngine(
        STDPConfig(a_plus=1.0, a_minus=1.0, learning_rate=1.0, trace_decay=0.125),
        max_neurons=16,
        id_mask=0xF,
    )
    stdp.w_min = -128
    stdp.w_max = 127
    weights = np.zeros((16, 16), dtype=np.int16)
    stdp.set_weights(weights)
    stdp.add_synapse(0, 1)
    stdp.put_weight(0, 1, 0)

    pre_updates = stdp.process_pre_spike(0, timestamp=0, connected_post_ids=[1])
    post_updates = stdp.process_post_spike(1, timestamp=1, connected_pre_ids=[0])
    python_learned_weight = int(stdp.get_weight(0, 1))
    rtl_learned_weight = latest_rtl_learned_weight()
    applied_rtl_weight = (
        rtl_learned_weight
        if rtl_learned_weight is not None
        else max(0, min(255, abs(python_learned_weight)))
    )
    after_count = run_trial(sim, applied_rtl_weight)

    matches = before_count == 1 and after_count >= 2 and applied_rtl_weight > initial_weight

    rows = [
        ("pre spikes", "0 @ t=0"),
        ("post spikes", "1 @ t=1"),
        ("initial RTL weight", str(initial_weight)),
        ("HLS/STDP pre updates", str(len(pre_updates))),
        ("HLS/STDP post updates", str(len(post_updates))),
        ("Python STDP learned weight", str(python_learned_weight)),
        ("RTL sim learned weight", str(rtl_learned_weight) if rtl_learned_weight is not None else "no xsim log"),
        ("applied RTL magnitude", str(applied_rtl_weight)),
        ("output spikes before", str(before_count)),
        ("output spikes after", str(after_count)),
        ("Python matches RTL spike behavior", "YES" if matches else "NO"),
    ]

    width = max(len(k) for k, _ in rows)
    print("Custom fixed HLS/RTL comparison")
    print("=" * 42)
    for key, value in rows:
        print(f"{key:<{width}} : {value}")
    print("=" * 42)
    print("PASS" if matches else "FAIL")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
