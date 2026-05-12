# MNIST Classifier Results

- Train images: 1000
- Test images: 200
- Input neurons: 64
- Output neurons: 10
- Encoding: 28x28 MNIST average-pooled to 8x8, contrast-normalized rate vector
- Accuracy: 0.5650
- Average input spikes/image: 18.57
- Average output spikes/image: 1.00
- Average latency: 18.57 cycles (0.186 us at 100 MHz)
- Throughput estimate: 5385029.62 images/s
- Assigned labels per output neuron: [1, 3, 7, 6, 7, 3, 8, 4, 1, 0]

## Weight Summary

- min: 0.000000
- max: 1.000000
- mean: 0.242471
- std: 0.347379
- mean_abs_delta: 0.229972
- max_abs_delta: 2.306196

## Per-Class Accuracy

- 0: 0.8824
- 1: 0.9286
- 2: 0.0000
- 3: 0.9375
- 4: 0.5357
- 5: 0.0000
- 6: 0.8500
- 7: 0.8333
- 8: 0.5000
- 9: 0.0000

## Confusion Matrix

Rows=true labels, columns=predicted labels.

| true\pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| 1 | 0 | 26 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2 | 0 | 0 | 0 | 9 | 0 | 0 | 5 | 1 | 1 | 0 |
| 3 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 1 | 0 |
| 4 | 1 | 3 | 0 | 0 | 15 | 0 | 2 | 7 | 0 | 0 |
| 5 | 1 | 1 | 0 | 11 | 2 | 0 | 0 | 1 | 4 | 0 |
| 6 | 2 | 0 | 0 | 0 | 0 | 0 | 17 | 1 | 0 | 0 |
| 7 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 20 | 1 | 0 |
| 8 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 5 | 0 |
| 9 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 14 | 3 | 0 |

## RTL Comparison

- 10: {'log': 'C:\\Users\\96898\\Desktop\\Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA\\hardware\\sim_work_rtl_mnist_classifier\\sim_tb_custom_rtl_mnist_classifier_10.log', 'train_images': 10, 'test_images': 10, 'accuracy': 0.1, 'avg_latency_cycles': 278, 'avg_latency_us_100mhz': 2.78, 'throughput_images_per_second_100mhz': 359712.23021582735, 'avg_output_spikes_per_image': 1.0, 'status': 'PASS'}
- 100: {'log': 'C:\\Users\\96898\\Desktop\\Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA\\hardware\\sim_work_rtl_mnist_classifier\\sim_tb_custom_rtl_mnist_classifier_100.log', 'train_images': 100, 'test_images': 100, 'accuracy': 0.25, 'avg_latency_cycles': 110, 'avg_latency_us_100mhz': 1.1, 'throughput_images_per_second_100mhz': 909090.9090909091, 'avg_output_spikes_per_image': 1.0, 'status': 'PASS'}
