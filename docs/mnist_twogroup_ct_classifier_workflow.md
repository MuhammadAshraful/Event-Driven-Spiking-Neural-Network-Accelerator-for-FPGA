# MNIST Two-Group CT Classifier Workflow

This milestone moves the custom MNIST classifier closer to the original
event-driven architecture by using:

- `event_router_ng`
- `synaptic_connectivity_table`
- two active `core_group` instances

No HLS AXI path, `snn_top_hls.v`, board wrapper, or `design_1_wrapper` is used.

## Architecture

Group 0 is the input group:

- local neuron IDs `0..63` receive MNIST spike events
- events arrive through `event_router_ng.ext_spike_*`
- group 0 neurons fire naturally from `core_group` LIF behavior

Group 1 is the output/classifier group:

- local neuron IDs `0..9` are the digit prototype neurons
- group 1 receives inter-group spikes through `event_router_ng`
- group 1 neurons fire naturally from `core_group` LIF behavior

The connectivity table stores 640 inter-group entries:

- source group: `0`
- source neuron: `0..63`
- fanout index: `0..9`
- destination group: `1`
- destination neuron: `0..9`
- excitatory valid weight

## Spike Flow

```text
MNIST event
  -> event_router_ng external route
  -> core_group 0 input neuron
  -> natural group 0 spike
  -> event_router_ng CT lookup
  -> synaptic_connectivity_table fanout
  -> event_router_ng inter-group delivery
  -> core_group 1 output neuron
  -> natural group 1 spike
  -> WTA winner
```

## Learning Flow

Training is unsupervised:

- labels are not used during training
- no teacher post-spikes are injected
- WTA selects the first natural group 1 output spike per image window
- active input neurons strengthen CT weights to the winning output neuron

Learned inter-group updates use:

- `learn_weight_valid`
- `learn_weight_group = 0`
- `learn_weight_src = active input neuron`
- `learn_weight_dst_group = 1`
- `learn_weight_dst = winning output neuron`
- `learn_weight_fanout_idx = winning output neuron`
- `learn_weight_is_inter = 1`

`event_router_ng` forwards these requests to `ct_cfg_*`, and
`synaptic_connectivity_table` performs the write. Direct CT/core learning
writes are intentionally avoided in the training path.

## What Remains Simplified

- Only two core groups are active.
- The input group uses one local spike per active MNIST event.
- WTA is still wrapper-level logic.
- Accuracy is low because the milestone prioritizes architecture fidelity.
- No 16-group partitioning or larger routing topology is used yet.

## Commands

```powershell
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_twogroup_ct_classifier_xsim.tcl -tclargs mnist10
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_twogroup_ct_classifier_xsim.tcl -tclargs mnist100
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_twogroup_ct_classifier_xsim.tcl -tclargs all
python software/python/custom_mnist_twogroup_ct_classifier_compare.py
```

Detailed results are written to:

- `outputs/mnist_twogroup_ct_classifier_results.md`
- `outputs/mnist_twogroup_ct_classifier_results.json`
