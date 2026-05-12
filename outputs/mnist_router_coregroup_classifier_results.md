# MNIST Router + Coregroup Classifier Results

## Architecture

- Active RTL core groups: 1
- event_router_ng instances: 1
- event_router_ng NUM_GROUPS: 2, with only group 0 active
- Input neurons: 64, local IDs 0..63
- Output neurons: 10, local IDs 64..73
- Input spikes enter through event_router_ng external-spike routing.
- Learned weights are issued through event_router_ng.learn_weight_*.
- Direct core_group learning writes are not used in this path.

## Python Reference

- Train images: 1000
- Test images: 200
- Accuracy: 0.2650
- Average input spikes/image: 18.86
- Average output spikes/image: 23.07
- Learned weight updates: 22783

## RTL Representative Runs

| subset | train | test | accuracy | latency cycles | latency us | throughput img/s | input spikes/img | output spikes/img | changed weights | router writes | routed spikes | observed spikes | direct writes | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 10 | 0.2000 | 552 | 5.520 | 180831 | 18.30 | 175.00 | 439 | 439 | 563 | 526 | 0 | PASS |
| 100 | 100 | 100 | 0.1400 | 552 | 5.520 | 180831 | 19.12 | 2493.51 | 540 | 540 | 5852 | 6299 | 0 | PASS |

## Python Confusion Matrix

Rows are true labels, columns are predicted labels.

| true\pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 12 | 0 |
| 1 | 0 | 12 | 0 | 1 | 0 | 0 | 0 | 0 | 15 | 0 |
| 2 | 0 | 1 | 2 | 5 | 0 | 0 | 2 | 1 | 5 | 0 |
| 3 | 0 | 0 | 5 | 9 | 0 | 0 | 0 | 0 | 2 | 0 |
| 4 | 0 | 9 | 0 | 7 | 0 | 0 | 0 | 0 | 12 | 0 |
| 5 | 0 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 6 | 0 |
| 6 | 0 | 0 | 10 | 0 | 0 | 0 | 10 | 0 | 0 | 0 |
| 7 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 11 | 8 | 0 |
| 8 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| 9 | 0 | 8 | 0 | 2 | 0 | 0 | 0 | 0 | 11 | 0 |

## Notes

- Labels are used only after unsupervised training for assignment/evaluation.
- The RTL shadow weight array is for logging/update bookkeeping only; spike processing uses core_group weight memory.
- synaptic_connectivity_table is intentionally not in this milestone yet.
