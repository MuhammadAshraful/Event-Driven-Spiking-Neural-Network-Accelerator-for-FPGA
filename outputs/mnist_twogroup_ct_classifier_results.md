# MNIST Two-Group CT Classifier Results

## Architecture

- Active core groups: 2
- Connectivity table: used for group 0 -> group 1 fanout
- CT entries used: 640
- Learned updates use `event_router_ng.learn_weight_*` with `learn_weight_is_inter=1`.
- Direct CT/core_group learning writes are avoided.

## Python Reference

- Train images: 1000
- Test images: 200
- Accuracy: 0.1200
- Average input spikes/image: 18.86
- Average group0 spikes/image: 18.86
- Average group1 spikes/image: 12.53
- Learned CT updates: 217

## RTL Representative Runs

| subset | train | test | accuracy | latency cycles | latency us | throughput img/s | input spikes/img | group0 spikes/img | group1 spikes/img | inter routed | CT updates | direct writes | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 10 | 0.2000 | 2217 | 22.170 | 45085 | 18.30 | 2.22 | 0.42 | 3814 | 155 | 0 | PASS |
| 100 | 100 | 100 | 0.1400 | 2217 | 22.170 | 45085 | 19.12 | 4.23 | 1.07 | 33852 | 191 | 0 | PASS |

## Python Confusion Matrix

Rows are true labels, columns are predicted labels.

| true\pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 |

## Notes

- Labels are used only after unsupervised training.
- Accuracy is still low because this milestone prioritizes architecture fidelity over tuning.
- The next step is scaling the same CT/router pattern to more core groups.
