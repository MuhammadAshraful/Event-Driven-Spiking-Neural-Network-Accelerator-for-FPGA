# Custom RTL STDP Learning Notes

This workflow adds a small RTL-only learning path. It intentionally avoids:

- HLS AXI-lite and AXI-stream control paths
- `snn_top_hls.v` as the main learning engine
- `design_1_wrapper`
- board deployment

## Direct RTL Weight Update Path

The cleanest available update path is already present in
`hardware/hdl/rtl/core/event_router_ng.v`.

The custom STDP engine drives:

- `learn_weight_valid`
- `learn_weight_group`
- `learn_weight_src`
- `learn_weight_dst`
- `learn_weight_data`
- `learn_weight_exc`
- `learn_weight_is_inter`
- `learn_weight_dst_group`
- `learn_weight_fanout_idx`

For the first experiment, `learn_weight_is_inter` is `0`, so
`event_router_ng` forwards the update to `core_group.weight_*`. The actual
weight memory write therefore still happens inside `core_group.v`; the custom
top does not poke internal memories.

## What The STDP Engine Observes

`custom_rtl_stdp_sim_top.v` routes testbench spikes through:

```text
testbench spike -> event_router_ng -> core_group
```

When a core group emits a spike, `event_router_ng` exposes it on:

- `learn_spike_valid`
- `learn_spike_src_id`
- `learn_spike_ready`

The custom top watches those learning notifications for group 0:

- neuron 0 is treated as the pre-synaptic neuron
- neuron 1 is treated as the post-synaptic neuron

The testbench can force a teacher post spike by injecting a normal external
spike into neuron 1. That spike is processed by `core_group` and observed by
the STDP engine through the same router learning notification path.

The custom top also has a `stdp_enable` input. The testbench keeps learning
disabled while it measures the before-learning behavior and while it clears the
teacher neuron state. Learning is enabled only for the intentional pre/post
training pair, then disabled again before the after-learning inference check.

## How Weight Changes Are Computed

`stdp_learning_engine.v` implements a simple pair-based STDP rule:

- If post fires after pre within `STDP_WINDOW`, apply LTP:
  `new_weight = min(current_weight + A_PLUS, W_MAX)`
- If pre fires after post within `STDP_WINDOW`, apply LTD:
  `new_weight = max(current_weight - A_MINUS, W_MIN)`
- Otherwise no update is emitted.

The module stores `last_pre_time[neuron]` and `last_post_time[neuron]`. For this
beginner-sized experiment it emits one update for the most recent observed
pre/post pair.

## How The Update Is Applied

The integration top tracks the demonstrated synapse `group0 neuron0 ->
group0 neuron1` in a small shadow register only so the STDP engine has a simple
`current_weight` input and the testbench has readable observability.

The real RTL write uses:

```text
stdp_learning_engine.update_* -> event_router_ng.learn_weight_* -> core_group.weight_*
```

For this fixed experiment:

- initial weight `n0 -> n1` is `5`
- threshold is `10`
- `A_PLUS` is `5`
- learned weight becomes `10`

One practical simulation detail: `core_group.v` initializes neuron state memory
in an `initial` block, but its reset path does not clear that RAM. A
sub-threshold before-learning trial can therefore leave membrane potential in
neuron 1. The integrated testbench intentionally forces neuron 1 to fire with
`stdp_enable=0` before the actual learning trial so the teacher neuron starts
the training pair from a clean membrane state.

## Files

- `hardware/hdl/rtl/learning/stdp_learning_engine.v`
- `hardware/hdl/tb/tb_stdp_learning_engine.v`
- `hardware/hdl/rtl/top/custom_rtl_stdp_sim_top.v`
- `hardware/hdl/tb/tb_custom_rtl_stdp_learning.v`
- `software/python/custom_rtl_stdp_compare.py`
- `hardware/scripts/run_custom_rtl_stdp_xsim.tcl`

## Run Commands

From a Vivado command shell:

```text
vivado -mode batch -source hardware/scripts/run_custom_rtl_stdp_xsim.tcl
vivado -mode batch -source hardware/scripts/run_custom_rtl_stdp_xsim.tcl -tclargs stdp
vivado -mode batch -source hardware/scripts/run_custom_rtl_stdp_xsim.tcl -tclargs custom
```

Python comparison:

```text
python software/python/custom_rtl_stdp_compare.py
```
