# MNIST Coregroup Classifier Results

## Python Reference

- Train images: 1000
- Test images: 200
- Input neurons: 64
- Output neurons: 10
- Accuracy: 0.2650
- Average input spikes/image: 18.86
- Average output spikes/image: 23.07
- Average latency cycles: 1.00
- Average latency at 100 MHz: 0.010 us
- Throughput estimate: 100000000.00 images/s
- Learned weight updates: 22783
- Assigned labels: [1, 0, 8, 6, 7, 3, 2, 0, 7, 2]

## RTL Representative Runs

| subset | train | test | accuracy | avg latency cycles | latency us | throughput img/s | avg input spikes | avg output spikes | updates | weights | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 10 | 10 | 10 | 0.2000 | 1511 | 15.110 | 66137 | 18.30 | 2.00 | 251 | 2/15/5466 | PASS |
| 100 | 100 | 100 | 0.0900 | 1541 | 15.410 | 64850 | 19.12 | 1.29 | 1637 | 0/15/5253 | PASS |

## Confusion Matrix

Python reference, rows=true labels, columns=predicted labels.

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

- RTL uses real `core_group` local weight memory for input-to-output synapses.
- The small RTL shadow weight array is bookkeeping only because `core_group` has no readback port.
- Labels are used only during post-training assignment and evaluation.
