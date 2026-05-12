# Unsupervised RTL MNIST Learning Notes

This workflow continues the custom RTL learning path and intentionally avoids:

- HLS AXI control and stream interfaces
- `snn_top_hls.v` as the learning engine
- `design_1_wrapper`
- board deployment
- teacher or forced post-spikes during unsupervised training

## What Changed From the Fixed Teacher Experiment

The earlier fixed STDP test used a controlled post spike to prove that the
learning engine and learned-weight write path worked. That was useful for
debugging, but it is not unsupervised learning.

The unsupervised path removes the teacher post-spike. The output neuron must now
fire because input spikes naturally excite it through RTL synaptic weights. The
custom STDP engine observes that natural output spike as the post spike.

## Signals Observed by STDP

The unsupervised simulation top watches the router learning-observation stream:

- pre spikes: natural spikes from input neurons 0, 1, and 2
- post spike: natural spike from output neuron 3

No separate teacher input is connected. If neuron 3 does not fire, no LTP update
is generated.

## Weight Update Path

The update path remains direct RTL:

```text
event_router_ng.learn_spike_* observation
    -> stdp_learning_engine
    -> event_router_ng.learn_weight_*
    -> core_group.weight_*
```

For the first staged experiments the learned intra-group paths are:

- neuron 0 -> neuron 3
- neuron 1 -> neuron 3
- neuron 2 -> neuron 3

The testbench can disable learning for inference by driving `stdp_enable=0`.
When learning is disabled, natural spikes still flow through the RTL SNN, but no
weight updates are accepted from the STDP engines.

## Minimal Fixed Experiment

The fixed unsupervised test uses one group with 16 physical neurons and only four
active logical roles:

- inputs: neurons 0, 1, 2
- output: neuron 3
- threshold: 10
- initial learned weights: 3, 3, 3

One input presentation is sub-threshold: `3 + 3 + 3 = 9`, so neuron 3 does not
fire before learning. During training, the same input pattern is repeated. The
residual membrane potential causes neuron 3 to fire naturally, and STDP
strengthens the active input-to-output weights. In inference, the same one-shot
input pattern is replayed with learning disabled; neuron 3 now fires.

## MNIST Smoke Workflow

Full 784-input MNIST RTL simulation would be slow and would require a larger
multi-output topology. The staged smoke test keeps RTL simulation practical by
pooling each image into three vertical input channels:

- left stroke energy -> neuron 0
- center stroke energy -> neuron 1
- right stroke energy -> neuron 2

The generator emits text files with this format:

```text
cycle neuron_id weight
```

Labels are written to a separate file for post-training evaluation only. The RTL
testbench does not read labels and does not use labels during training.

## Winner-Take-All Status

The current MNIST smoke test uses one output neuron, so there is no competition
yet. A small reusable `winner_take_all.v` helper is included for the next
multi-output stage, where output neurons can compete and only the first winner
inside a short inhibition window should receive credit.

## Simplifications

This is an unsupervised RTL learning workflow, not a complete biological SNN and
not a full MNIST classifier yet. The current simplifications are:

- 3 pooled input channels instead of 784 pixel neurons
- 1 output neuron in the first MNIST smoke stage
- additive pair-based LTP only in the unsupervised demo (`A_MINUS=0`)
- no homeostasis, adaptive threshold, or lateral inhibition in the first smoke
  run
- RTL validation is staged on 1 image, 10 images, and a small batch rather than
  60,000 images

The important result is that MNIST-derived spike events drive the RTL subsystem,
natural post spikes trigger STDP, learned weights are written back through the
router into `core_group`, and inference runs with learning disabled.

## Run Commands

Generate the staged MNIST spike files:

```text
python software/python/generate_mnist_spike_files.py
```

Run the unsupervised RTL xsim targets:

```text
vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs unsup_fixed
vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs mnist1
vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs mnist10
vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs batch
vivado -mode batch -source hardware/scripts/run_custom_rtl_unsupervised_xsim.tcl -tclargs all
```

Compare the RTL logs with the Python reference:

```text
python software/python/custom_unsupervised_mnist_compare.py
```

The result table is written to:

```text
outputs/unsupervised_mnist_results.md
```
