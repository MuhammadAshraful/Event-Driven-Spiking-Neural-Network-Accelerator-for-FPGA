# MNIST Router + Coregroup Classifier Workflow

This workflow is a staged step from the direct classifier toward the original
router/core-group architecture. It keeps the working one-core `core_group`
classifier intact and adds a new path that places `event_router_ng` around it.

## What This Path Uses

- One real `core_group` instance with 128 physical LIF neurons.
- Logical input neurons: local IDs `0..63`.
- Logical output neurons: local IDs `64..73`.
- One `event_router_ng` instance configured with `NUM_GROUPS=2`.
- Only router group 0 is active.
- Group 1 exists only because `NUM_GROUPS=1` creates a zero-width group ID in
  the current router module.

## Spike Path

MNIST spike events are not driven directly into `core_group`.

The testbench sends each event to the classifier top. The top queues one event
at a time and drives:

- `event_router_ng.ext_spike_valid`
- `event_router_ng.ext_spike_neuron_id`
- `event_router_ng.ext_spike_weight`
- `event_router_ng.ext_spike_exc`

The router then drives group-0 input ports:

- `grp_in_valid[0]`
- `grp_in_dest_id[0]`
- `grp_in_weight[0]`
- `grp_in_exc[0]`

Those ports connect to the real `core_group.ext_spike_*` inputs.

## Natural Output Spikes

Output neurons fire from normal `core_group` LIF behavior:

- membrane accumulation
- threshold comparison
- refractory handling
- local input-to-output weights in `core_group` memory

The wrapper captures each `core_group` output spike once and forwards it into
the router group-spike input. WTA/STDP observes only natural output spikes; no
teacher post-spikes are used.

## Learned Weight Path

Learning updates no longer write directly from the classifier FSM into
`core_group.weight_*`.

Instead, the classifier emits intra-group updates through:

- `learn_weight_valid`
- `learn_weight_group = 0`
- `learn_weight_src`
- `learn_weight_dst`
- `learn_weight_data`
- `learn_weight_exc`
- `learn_weight_is_inter = 0`

`event_router_ng` forwards those updates to:

- `grp_weight_we[0]`
- `grp_weight_src`
- `grp_weight_dst`
- `grp_weight_data`
- `grp_weight_exc`

Those forwarded signals connect to `core_group.weight_*`.

The local `weight_shadow` array remains only for logging and update
bookkeeping, because `core_group` has no weight readback port. Spike processing
uses the actual `core_group` weight memory.

## What Is Still Simplified

- `synaptic_connectivity_table` is not used yet.
- There is only one active `core_group`.
- Inter-group fanout is not exercised.
- WTA is still a small wrapper-level mechanism.
- Accuracy is not optimized; this milestone proves the router-mediated spike
  and learned-weight paths.

## Run Commands

```powershell
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_router_coregroup_classifier_xsim.tcl -tclargs mnist10
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_router_coregroup_classifier_xsim.tcl -tclargs mnist100
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_router_coregroup_classifier_xsim.tcl -tclargs all
python software/python/custom_mnist_router_coregroup_classifier_compare.py
```

## Current Verified Results

- `mnist10`: RTL PASS, router external spike path used, router learned-weight
  path used, direct learning writes avoided.
- `mnist100`: RTL PASS, router external spike path used, router learned-weight
  path used, direct learning writes avoided.

Detailed metrics are written to:

- `outputs/mnist_router_coregroup_classifier_results.md`
- `outputs/mnist_router_coregroup_classifier_results.json`
