# MNIST Two-Group CT Classifier Results

## Architecture

- Active core groups: 2
- Connectivity table: used for group 0 -> group 1 fanout
- CT entries used: 640
- Learned updates use `event_router_ng.learn_weight_*` with `learn_weight_is_inter=1`.
- Direct CT/core_group learning writes are avoided.
- WTA: window-level winner from natural group-1 spikes.
- Homeostasis: training winner selection favors least-used spiking outputs.
- LTD/normalization: overused active outputs can be depressed; incoming row sum is capped.

## Python Reference

- Train images: 1000
- Test images: 200
- Accuracy: 0.0850
- Assignment winner counts: [14, 4, 29, 6, 7, 0, 0, 1, 3, 0]
- Dead outputs: 3
- Dominant ratio: 0.029
- Average input spikes/image: 18.86
- Average group0 spikes/image: 18.86
- Average group1 spikes/image: 0.05
- Learned CT updates: 19535

## RTL Representative Runs

| subset | train | test | accuracy | latency cycles | latency us | throughput img/s | input spikes/img | group0 spikes/img | group1 spikes/img | inter routed | CT updates | dead outputs | dominant ratio | direct writes | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 10 | 0.3000 | 631 | 6.310 | 158227 | 18.30 | 1.07 | 0.03 | 5413 | 5285 | 3 | 0.200 | 0 | PASS |

RTL `10` assignment winner counts: [1, 0, 1, 1, 1, 2, 2, 2, 0, 0]
RTL `10` assigned labels: [1, 0, 2, 5, 0, 3, 1, 1, 0, 0]
RTL `10` training winner counts: [2, 0, 1, 0, 0, 3, 2, 2, 0, 0]

| 100 | 100 | 100 | 0.1400 | 631 | 6.310 | 158227 | 19.12 | 5.12 | 0.01 | 57589 | 5975 | 5 | 0.200 | 0 | PASS |

RTL `100` assignment winner counts: [0, 20, 0, 0, 20, 0, 20, 20, 0, 20]
RTL `100` assigned labels: [0, 6, 0, 0, 1, 0, 3, 0, 0, 1]
RTL `100` training winner counts: [5, 15, 3, 2, 17, 6, 17, 17, 0, 18]

## Python Confusion Matrix

Rows are true labels, columns are predicted labels.

| true\pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| 8 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Notes

- Labels are used only after unsupervised training for output-neuron assignment.
- Accuracy is still low; this pass mainly improves output-neuron usage and reduces one-neuron collapse.
- The next tuning target is better prototype formation before scaling to four or more groups.
