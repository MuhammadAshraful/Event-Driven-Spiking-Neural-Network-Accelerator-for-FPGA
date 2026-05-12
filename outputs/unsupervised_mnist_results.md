# Unsupervised MNIST RTL Results

| experiment | input spikes | input neurons used | output neurons used | training mode | before-learning output spikes | after-learning output spikes | learned weight changes | winner neuron | RTL PASS/FAIL | Python comparison PASS/FAIL | simulation time |
|---|---:|---|---:|---|---:|---:|---|---:|---|---|---|
| mnist1 | 9 | 3 pooled | 1 | unsupervised STDP | 1 | 1 | 7/7/7 | 3 | PASS | PASS | not captured |
| mnist10 | 79 | 3 pooled | 1 | unsupervised STDP | 7 | 10 | 15/15/15 | 3 | PASS | PASS | not captured |
| batch | 196 | 3 pooled | 1 | unsupervised STDP | 16 | 25 | 15/15/15 | 3 | PASS | PASS | not captured |

Notes:
- Labels are not used by RTL training; they are emitted separately by the spike generator for later evaluation.
- This smoke test pools each 28x28 image into three input neurons and one output neuron.
- Python comparison allows a small spike-count tolerance because RTL handshakes add cycle spacing around each event.
