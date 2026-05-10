#!/usr/bin/env python
"""Compare the custom RTL-only STDP experiment against Python behavior."""

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


RTL_LOG = REPO_ROOT / "hardware" / "sim_work_rtl_stdp" / "sim_tb_custom_rtl_stdp_learning.log"


def parse_rtl_log() -> dict[str, int] | None:
    if not RTL_LOG.exists():
        return None

    text = RTL_LOG.read_text(errors="ignore")

    def last_int(pattern: str) -> int | None:
        matches = re.findall(pattern, text)
        return int(matches[-1]) if matches else None

    before_after = re.findall(r"output spikes before/after\s*:\s*(\d+)\s*/\s*(\d+)", text)
    result = {
        "initial_weight": last_int(r"initial weight\s*:\s*(\d+)"),
        "updated_weight": last_int(r"updated weight\s*:\s*(\d+)"),
        "rule": last_int(r"STDP rule applied\s*:\s*(\d+)"),
    }
    if before_after:
        result["before_spikes"] = int(before_after[-1][0])
        result["after_spikes"] = int(before_after[-1][1])

    return result if all(value is not None for value in result.values()) else None


def run_trial(weight_0_1: int) -> int:
    params = LIFNeuronParams(threshold=10, leak_rate=0, refractory_period=3)
    sim = HWAccurateSNNSimulator(
        num_groups=2,
        neurons_per_group=16,
        neuron_params=params,
    )
    sim.reset()
    sim.set_intra_weight(0, 0, 1, weight_0_1, exc=True)
    result = sim.run(
        30,
        input_spike_train={0: [(sim.local_to_global(0, 0), 12, True)]},
    )
    return int(result["per_group_spikes"][0])


def main() -> int:
    initial_weight = 5
    a_plus = 5
    w_max = 15
    updated_weight = min(initial_weight + a_plus, w_max)

    before_spikes = run_trial(initial_weight)
    after_spikes = run_trial(updated_weight)

    rtl = parse_rtl_log()
    rtl_before = rtl.get("before_spikes") if rtl else None
    rtl_after = rtl.get("after_spikes") if rtl else None
    rtl_updated = rtl.get("updated_weight") if rtl else None

    python_ok = before_spikes == 1 and after_spikes >= 2 and updated_weight >= 10
    rtl_ok = (
        rtl is not None
        and rtl_before == before_spikes
        and rtl_after == after_spikes
        and rtl_updated == updated_weight
    )

    rows = [
        ("initial weight", str(initial_weight)),
        ("updated weight", str(updated_weight)),
        ("before-learning Python spikes", str(before_spikes)),
        ("after-learning Python spikes", str(after_spikes)),
        ("RTL updated weight", str(rtl_updated) if rtl_updated is not None else "no xsim log"),
        ("RTL before-learning spikes", str(rtl_before) if rtl_before is not None else "no xsim log"),
        ("RTL after-learning spikes", str(rtl_after) if rtl_after is not None else "no xsim log"),
        ("Python expectation", "PASS" if python_ok else "FAIL"),
        ("Python matches RTL", "YES" if rtl_ok else "NO"),
    ]

    width = max(len(key) for key, _ in rows)
    print("Custom RTL STDP comparison")
    print("=" * 42)
    for key, value in rows:
        print(f"{key:<{width}} : {value}")
    print("=" * 42)
    print("PASS" if python_ok and rtl_ok else "FAIL")
    return 0 if python_ok and rtl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
