# MNIST Coregroup Classifier Workflow

This path refactors the previous direct 64x10 classifier toward the intended RTL
architecture. It does not delete or replace the working direct classifier.

## What This Path Uses

- one real `core_group.v` instance
- 128 physical LIF neurons
- input neurons: 0..63
- output neurons: 64..73
- real `core_group` local weight memory for input-to-output synapses
- natural LIF output spikes from `core_group`
- WTA over output-neuron spikes
- STDP-style updates written back through `core_group.weight_*`

The small shadow weight array in
`custom_rtl_mnist_coregroup_classifier_top.v` exists only because `core_group`
does not provide a weight readback port. Spike processing uses the real
`core_group` weight RAM.

## What This Path Avoids

- no `design_1_wrapper`
- no HLS AXI
- no `snn_top_hls.v`
- no labels during training
- no teacher post-spikes

## Training And Evaluation

The testbench runs three phases:

1. Training: MNIST spikes drive input neurons, output neurons fire naturally,
   WTA picks the first winner, and active input-to-winner weights are updated.
2. Label assignment: training images are replayed with learning disabled. Labels
   are used only now to map output neurons to digit classes.
3. Test: test images are replayed with learning disabled. Prediction is the
   assigned label of the winning output neuron.

## Event Files

This path reuses the 8x8 classifier files from:

```text
python software/python/generate_mnist_classifier_files.py
```

Event format:

```text
cycle neuron_id weight image_id phase
```

## Run Commands

```text
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_coregroup_classifier_xsim.tcl -tclargs mnist10
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_coregroup_classifier_xsim.tcl -tclargs mnist100
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_coregroup_classifier_xsim.tcl -tclargs all
python software/python/custom_mnist_coregroup_classifier_compare.py
```

## Current Results

Latest representative RTL results:

- `mnist10`: PASS, 20.0% accuracy, average latency 1511 cycles
- `mnist100`: PASS, 9.0% accuracy, average latency 1541 cycles

Python coregroup-like reference:

- train images: 1000
- test images: 200
- accuracy: 26.5%

Result files:

- `outputs/mnist_coregroup_classifier_results.md`
- `outputs/mnist_coregroup_classifier_results.json`

## Honest Limitations

- This is now much more architecture-faithful than the direct dense classifier,
  but it still uses one `core_group` only.
- `event_router_ng` and `synaptic_connectivity_table` are not yet in the MNIST
  classifier path.
- Accuracy is currently low because all 10 output neurons are single prototypes
  sharing one global threshold/refractory setting, and learning is intentionally
  simple.
- The next architectural step is to split inputs/outputs across multiple
  `core_group`s and route input fanout through `event_router_ng` + CT, while
  keeping learned updates on the router `learn_weight_*` path.
