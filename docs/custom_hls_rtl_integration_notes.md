# Custom HLS/RTL Simulation Integration Notes

This note documents the checked-in generated HLS RTL at:

`hardware/hls/hls_csim_output/hls/impl/verilog/snn_top_hls.v`

The goal of the custom simulation workflow is to use this generated Verilog
directly, without `design_1_wrapper` and without a board.

## Generated HLS Module Ports

The generated top module is:

```verilog
module snn_top_hls (
    ap_clk,
    ap_rst_n,
    ...
);
```

### Clock, Reset, Control

- `ap_clk`: kernel clock.
- `ap_rst_n`: active-low reset.
- `interrupt`: AXI-lite interrupt output.
- The HLS `ap_ctrl_hs` signals are not top-level ports. They are controlled
  through the `s_axi_ctrl` AXI-lite slave.

AXI-lite control map from `snn_top_hls_ctrl_s_axi.v`:

- `0x00`: HLS control register
  - bit 0: `ap_start`
  - bit 1: `ap_done`, clear on read
  - bit 2: `ap_idle`
  - bit 3: `ap_ready`, clear on read
  - bit 7: `auto_restart`
  - bit 9: `interrupt`
- `0x10`: `ctrl_reg`
  - bit 0: SNN enable
  - bit 1: SNN reset
  - bit 2: clear counters
  - bit 3: learning enable
  - bit 4: weight read mode
  - bit 5: apply reward
  - bit 6: weight load mode
  - bit 7: first spike only
- `0x18`: `config_reg`
  - `[15:0]`: threshold
  - `[31:16]`: leak rate
- `0x20`: `mode_reg`
  - `[1:0]`: operation mode, generated RTL uses `0=inference`, `1=STDP train`, `2=checkpoint`
  - bit 8: encoder enable
- `0x28`: `time_steps_reg`
- `0x30..0x40`: packed `learning_params`
- `0x48..0x58`: packed `encoder_config`
- `0x60`: `status_reg`
- `0x70`: `spike_count_reg`
- `0x80`: `weight_sum_reg`
- `0x90`: `version_reg`
- `0xa0`: `reward_signal`

`ap_start` is asserted by writing bit 0 of address `0x00`. `ap_done` is
latched by the AXI-lite control block and can be polled at address `0x00`.

### AXI-Stream Ports

Input streams:

- `s_axis_spikes_*`: spike/event input from a testbench or simulated host.
- `s_axis_data_*`: raw frame input for the HLS encoder path.
- `s_axis_weights_*`: weight load input.

Output streams:

- `m_axis_spikes_*`: spike/event output to a testbench or simulated host.
- `m_axis_weights_*`: weight checkpoint/debug output.

For the checked-in generated RTL, the useful packet layouts are:

- `s_axis_spikes_TDATA`
  - `[7:0]`: neuron id
  - `[9:8]`: unused in this generated wrapper
  - `[17:10]`: signed/legacy spike weight
- `s_axis_weights_TDATA`
  - `[7:0]`: pre neuron id
  - `[15:8]`: post neuron id
  - `[23:16]`: weight
  - generated storage uses `post_id[0]` as a RAM-bank select and
    `{pre_id, post_id[7:1]}` as the RAM address.
- `m_axis_weights_TDATA`
  - `[7:0]`: zero in this generated wrapper
  - `[15:8]`: checkpoint column/index byte
  - `[23:16]`: weight byte read from the selected HLS weight RAM

### Direct RTL Spike Handshake Ports

These can be connected directly to RTL/router logic:

- HLS to RTL:
  - `spike_in_valid`
  - `spike_in_neuron_id[7:0]`
  - `spike_in_weight[7:0]`
  - `spike_in_ready`
- RTL to HLS:
  - `spike_out_valid`
  - `spike_out_neuron_id[7:0]`
  - `spike_out_weight[7:0]`
  - `spike_out_ready`
- HLS control/status:
  - `snn_enable`
  - `snn_reset`
  - `threshold_out`
  - `leak_rate_out`
  - `snn_ready`
  - `snn_busy`

The generated direct outputs are `ap_none` style. In simulation they are best
treated as cycle-level wires driven during a kernel invocation. The testbench
still must start the HLS kernel through AXI-lite.

The custom integration top does not pass `spike_in_valid` straight into the
router because the generated `ap_none` valid can stay asserted for many cycles
or across kernel starts. Instead, it forwards one router event when the
generated HLS AXI-stream input accepts a spike (`s_axis_spikes_TVALID &&
s_axis_spikes_TREADY`). The standalone HLS testbench still observes the direct
`spike_in_*` outputs to prove the generated wrapper emits them.

### Learning / Weight Update Ports

Important blocker:

- `hardware/hls/src/snn_top_hls.cpp` and
  `hardware/hls/include/snn_top_hls.h` declare direct learned-weight ports:
  `learn_weight_valid`, `learn_weight_group`, `learn_weight_src`,
  `learn_weight_dst`, `learn_weight_data`, `learn_weight_exc`,
  `learn_weight_is_inter`, `learn_weight_dst_group`,
  `learn_weight_fanout_idx`, and `learn_weight_ready`.
- The checked-in generated Verilog
  `hardware/hls/hls_csim_output/hls/impl/verilog/snn_top_hls.v` does not expose
  any `learn_weight_*` ports.

That means the generated HLS Verilog cannot be directly wired to
`event_router_ng`'s learned-weight update port. For simulation, the custom top
uses a testbench adapter:

1. Run the generated HLS learning engine.
2. Request HLS weight checkpoint output on `m_axis_weights`.
3. Decode the relevant checkpoint word.
4. Forward that decoded weight into `event_router_ng.learn_weight_*`.

To use the newer direct learned-weight bridge, regenerate the HLS Verilog from
the current C++ source and replace/update the generated files under
`hardware/hls/hls_csim_output/hls/impl/verilog/`.

## What Can Be Connected Directly

Direct or nearly-direct connections:

- `ap_clk`, `ap_rst_n`.
- HLS-accepted `s_axis_spikes` events to `event_router_ng.ext_spike_*` through
  a small one-event adapter and signed-weight to RTL magnitude/excitatory
  conversion.
- `snn_enable`, `snn_reset`, `threshold_out`, and `leak_rate_out` to RTL
  control, if the testbench wants HLS-driven control.
- `snn_ready` and `snn_busy` from RTL/router status.
- AXI-stream `s_axis_spikes` and AXI-lite `s_axi_ctrl` from the testbench.

Needs a testbench driver or adapter:

- All `s_axi_ctrl_*` ports require an AXI-lite master driver.
- All AXI-stream inputs require valid/ready source drivers.
- `m_axis_weights` must be decoded to obtain generated-HLS learned weights,
  because the generated Verilog lacks the newer direct `learn_weight_*` bridge.
- Direct `spike_in_*` requires pulse/transaction adaptation before driving RTL
  FIFOs, because the checked-in generated `ap_none` valid is level-style rather
  than a one-cycle event.
- `spike_out_*` can be connected to RTL output spike observations, but in the
  small deterministic experiment a testbench "teacher post spike" is also used
  so the HLS STDP path receives a known pre-then-post pair.

## Added Simulation Files

- `hardware/hdl/tb/tb_hls_learning_engine.v`: standalone generated-HLS testbench.
- `hardware/hdl/rtl/top/custom_hls_rtl_sim_top.v`: simulation-only HLS + RTL top.
- `hardware/hdl/tb/tb_custom_hls_rtl_learning.v`: first-principles integrated
  learning experiment.
- `software/python/custom_fixed_hls_rtl_compare.py`: Python comparison for the
  same small experiment.
- `hardware/hdl/tb/tb_mnist_hls_rtl_sim.v`: one-image MNIST-style spike-file
  smoke test through the same HLS/RTL top.
- `hardware/hdl/tb/data/mnist_one_image_spikes.mem`: tiny `cycle neuron_id
  weight` event file from a 4x4 pooled MNIST test image.
- `software/python/custom_mnist_hls_rtl_compare.py`: Python comparison for the
  one-image spike-file smoke test.
- `hardware/scripts/run_custom_hls_rtl_xsim.tcl`: xsim compile/elaborate/run
  command script.

## Run Commands

From a Vivado command shell:

```text
vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl
vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl -tclargs hls
vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl -tclargs custom
vivado -mode batch -source hardware/scripts/run_custom_hls_rtl_xsim.tcl -tclargs mnist
```

Python comparison commands:

```text
python software/python/custom_fixed_hls_rtl_compare.py
python software/python/custom_mnist_hls_rtl_compare.py
```
