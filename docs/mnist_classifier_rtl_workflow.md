# RTL/Python Unsupervised MNIST Classifier Workflow

This workflow is separate from the earlier 3-input smoke test. It keeps the
same constraints:

- no HLS AXI
- no `snn_top_hls.v`
- no `design_1_wrapper`
- no labels during training

## Architecture

The classifier uses:

- 64 input neurons from an 8x8 average-pooled MNIST image
- 10 output neurons
- winner-take-all output competition
- winner-only STDP-style learning
- post-training label assignment

Python is the large-scale reference. It trains on 1000 MNIST images by default
and evaluates on 200 test images. RTL is used for representative validation on
10 and 100 image subsets because full RTL simulation of 1000 images is slower.

## Python Reference

`software/python/custom_mnist_classifier_reference.py` implements a compact
competitive-STDP model:

1. Downsample 28x28 images to 8x8.
2. Normalize each image into a 64-value rate vector.
3. Pick the winning output neuron by membrane score minus adaptive threshold.
4. Update only the winner's weights toward the active input pattern.
5. Apply a small adaptive-threshold homeostasis term so one neuron does not
   dominate all images.
6. After training, replay training images and assign each output neuron to the
   digit label it most often wins.
7. Replay test images and calculate accuracy and metrics.

Labels are loaded only for steps 6 and 7.

## RTL Representative Classifier

`hardware/hdl/rtl/top/custom_rtl_mnist_classifier_top.v` contains a direct
64x10 dense weight matrix for simulation practicality. It also instantiates:

- `core_group.v` as the accepted-input spike front-end
- `winner_take_all.v` for output winner selection

The RTL testbench performs three passes:

1. training pass with `learning_enable=1`
2. label-assignment pass with `learning_enable=0`
3. test pass with `learning_enable=0`

The testbench logs winners, assigned labels, accuracy, average latency, average
output spikes per image, weight range/sum, and total weight updates.

## Event Files

Generate classifier files with:

```text
python software/python/generate_mnist_classifier_files.py
```

The generated event format is:

```text
cycle neuron_id weight image_id phase
```

where `phase=0` is training and `phase=1` is test/inference. Labels are stored
in separate files and are not used by RTL training.

## Run Commands

Python reference:

```text
python software/python/custom_mnist_classifier_reference.py
```

RTL representative validation:

```text
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_classifier_xsim.tcl -tclargs mnist10
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_classifier_xsim.tcl -tclargs mnist100
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_classifier_xsim.tcl -tclargs all
```

Optional 1000-image RTL target exists, but it is expected to be slower:

```text
vivado -mode batch -source hardware/scripts/run_custom_rtl_mnist_classifier_xsim.tcl -tclargs mnist1000
```

## Current Results

The latest result files are:

- `outputs/mnist_classifier_results.md`
- `outputs/mnist_classifier_results.json`

Current Python reference result:

- train images: 1000
- test images: 200
- accuracy: 56.5%
- average latency estimate: 18.57 cycles
- throughput estimate at 100 MHz: about 5.39 million images/s

Current RTL representative validation:

- 10 train / 10 test: PASS, 10.0% accuracy
- 100 train / 100 test: PASS, 25.0% accuracy

The RTL accuracy is lower than Python because the RTL validation uses a simpler
integer STDP update and much smaller training subsets. It still verifies the
end-to-end unsupervised classifier workflow: train without labels, assign labels
after training, test, and report metrics.

## Limitations and Next Improvements

- The Python model is still a compact 10-prototype unsupervised classifier, not
  a high-accuracy supervised MNIST model.
- The RTL classifier uses a direct 64x10 matrix for simulation practicality.
  Mapping the full dense matrix into `core_group` recurrent weights with a
  cycle-by-cycle hardware update sequencer is the next integration step.
- Accuracy can improve with more output neurons, multiple prototypes per digit,
  stronger homeostasis, and longer training.
