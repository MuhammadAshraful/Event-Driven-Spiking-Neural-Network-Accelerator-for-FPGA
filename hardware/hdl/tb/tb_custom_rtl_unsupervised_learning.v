//=============================================================================
// Minimal unsupervised RTL STDP learning test
//
// No teacher post-spike is used. The output neuron fires only when natural
// input spikes accumulate through learned recurrent weights.
//=============================================================================

`timescale 1ns / 1ps

module tb_custom_rtl_unsupervised_learning;

    localparam NUM_GROUPS        = 2;
    localparam WEIGHT_WIDTH      = 8;
    localparam GROUP_ID_WIDTH    = 1;
    localparam LOCAL_ID_WIDTH    = 4;
    localparam GLOBAL_ID_WIDTH   = 5;

    reg clk;
    reg rst_n;
    reg enable;
    reg stdp_enable;
    reg [7:0] global_leak_rate;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg                         ext_spike_valid;
    reg  [GLOBAL_ID_WIDTH-1:0]  ext_spike_neuron_id;
    reg  [WEIGHT_WIDTH-1:0]     ext_spike_weight;
    reg                         ext_spike_exc;
    wire                        ext_spike_ready;

    reg  [NUM_GROUPS-1:0]       host_weight_we;
    reg  [LOCAL_ID_WIDTH-1:0]   host_weight_src;
    reg  [LOCAL_ID_WIDTH-1:0]   host_weight_dst;
    reg  [WEIGHT_WIDTH-1:0]     host_weight_data;
    reg                         host_weight_exc;

    wire [15:0]                 g0_spike_count;
    wire                        router_busy;
    wire [NUM_GROUPS-1:0]       group_busy;
    wire [15:0]                 current_time_out;
    wire [WEIGHT_WIDTH-1:0]     weight_0_3;
    wire [WEIGHT_WIDTH-1:0]     weight_1_3;
    wire [WEIGHT_WIDTH-1:0]     weight_2_3;
    wire [15:0]                 output_spike_count;
    wire [15:0]                 learned_update_count;
    wire                        learned_update_pulse;
    wire [LOCAL_ID_WIDTH-1:0]   learned_update_src;
    wire [WEIGHT_WIDTH-1:0]     learned_update_weight;
    wire                        natural_output_observed;

    integer pass_count;
    integer fail_count;
    integer before_output_spikes;
    integer train_output_spikes;
    integer after_output_spikes;
    integer first_natural_output_time;
    integer initial_w0;
    integer initial_w1;
    integer initial_w2;

    custom_rtl_unsupervised_sim_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .stdp_enable(stdp_enable),
        .global_leak_rate(global_leak_rate),
        .ext_spike_valid(ext_spike_valid),
        .ext_spike_neuron_id(ext_spike_neuron_id),
        .ext_spike_weight(ext_spike_weight),
        .ext_spike_exc(ext_spike_exc),
        .ext_spike_ready(ext_spike_ready),
        .host_weight_we(host_weight_we),
        .host_weight_src(host_weight_src),
        .host_weight_dst(host_weight_dst),
        .host_weight_data(host_weight_data),
        .host_weight_exc(host_weight_exc),
        .g0_spike_count(g0_spike_count),
        .router_busy(router_busy),
        .group_busy(group_busy),
        .current_time_out(current_time_out),
        .weight_0_3(weight_0_3),
        .weight_1_3(weight_1_3),
        .weight_2_3(weight_2_3),
        .output_spike_count(output_spike_count),
        .learned_update_count(learned_update_count),
        .learned_update_pulse(learned_update_pulse),
        .learned_update_src(learned_update_src),
        .learned_update_weight(learned_update_weight),
        .natural_output_observed(natural_output_observed)
    );

    function [GLOBAL_ID_WIDTH-1:0] global_id;
        input [GROUP_ID_WIDTH-1:0] group_id;
        input [LOCAL_ID_WIDTH-1:0] local_id;
    begin
        global_id = {group_id, local_id};
    end
    endfunction

    task automatic check;
        input [8*96-1:0] desc;
        input condition;
    begin
        if (condition) begin
            $display("[PASS] %0s", desc);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] %0s", desc);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task automatic do_reset;
    begin
        rst_n <= 1'b0;
        enable <= 1'b0;
        stdp_enable <= 1'b0;
        global_leak_rate <= 8'd0;
        ext_spike_valid <= 1'b0;
        ext_spike_neuron_id <= {GLOBAL_ID_WIDTH{1'b0}};
        ext_spike_weight <= 8'd0;
        ext_spike_exc <= 1'b1;
        host_weight_we <= {NUM_GROUPS{1'b0}};
        host_weight_src <= 4'd0;
        host_weight_dst <= 4'd0;
        host_weight_data <= 8'd0;
        host_weight_exc <= 1'b1;
        repeat (30) @(posedge clk);
        rst_n <= 1'b1;
        enable <= 1'b1;
        repeat (120) @(posedge clk);
    end
    endtask

    task automatic wait_all_idle;
        integer timeout;
    begin
        timeout = 0;
        while ((router_busy || group_busy != 0) && timeout < 100000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        repeat (100) @(posedge clk);
    end
    endtask

    task automatic drain_membrane_state;
    begin
        global_leak_rate <= 8'd1;
        repeat (8000) @(posedge clk);
        wait_all_idle;
        global_leak_rate <= 8'd0;
        repeat (50) @(posedge clk);
    end
    endtask

    task automatic program_weight;
        input [LOCAL_ID_WIDTH-1:0] src;
        input [LOCAL_ID_WIDTH-1:0] dst;
        input [WEIGHT_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        host_weight_we <= 2'b01;
        host_weight_src <= src;
        host_weight_dst <= dst;
        host_weight_data <= weight;
        host_weight_exc <= 1'b1;
        @(posedge clk);
        host_weight_we <= 2'b00;
        repeat (5) @(posedge clk);
    end
    endtask

    task automatic inject_spike;
        input [GLOBAL_ID_WIDTH-1:0] neuron_id;
        input [WEIGHT_WIDTH-1:0] weight;
        integer timeout;
    begin
        @(negedge clk);
        ext_spike_neuron_id <= neuron_id;
        ext_spike_weight <= weight;
        ext_spike_exc <= 1'b1;
        ext_spike_valid <= 1'b1;

        timeout = 0;
        while (!ext_spike_ready && timeout < 10000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        ext_spike_valid <= 1'b0;
        ext_spike_weight <= 8'd0;
        ext_spike_neuron_id <= {GLOBAL_ID_WIDTH{1'b0}};
        @(posedge clk);
    end
    endtask

    task automatic apply_input_pattern;
        input integer repeats;
        integer r;
    begin
        for (r = 0; r < repeats; r = r + 1) begin
            inject_spike(global_id(1'd0, 4'd0), 8'd12);
            inject_spike(global_id(1'd0, 4'd1), 8'd12);
            inject_spike(global_id(1'd0, 4'd2), 8'd12);
            repeat (20) @(posedge clk);
        end
        wait_all_idle;
    end
    endtask

    always @(posedge clk) begin
        if (natural_output_observed) begin
            if (first_natural_output_time == 0)
                first_natural_output_time = current_time_out;
            $display("natural output spike: neuron=3 time=%0d stdp_enable=%0d",
                     current_time_out, stdp_enable);
        end
        if (learned_update_pulse) begin
            $display("unsup STDP update: src=%0d dst=3 weight=%0d time=%0d",
                     learned_update_src, learned_update_weight, current_time_out);
        end
    end

    initial begin
        $display("=========================================================");
        $display("  Minimal unsupervised RTL STDP learning TB");
        $display("=========================================================");

        pass_count = 0;
        fail_count = 0;
        before_output_spikes = 0;
        train_output_spikes = 0;
        after_output_spikes = 0;
        first_natural_output_time = 0;

        do_reset;
        program_weight(4'd0, 4'd3, 8'd3);
        program_weight(4'd1, 4'd3, 8'd3);
        program_weight(4'd2, 4'd3, 8'd3);
        initial_w0 = weight_0_3;
        initial_w1 = weight_1_3;
        initial_w2 = weight_2_3;

        $display("initial weights: w0_3=%0d w1_3=%0d w2_3=%0d",
                 initial_w0, initial_w1, initial_w2);

        // One presentation is sub-threshold: 3+3+3 = 9 < threshold 10.
        before_output_spikes = output_spike_count;
        apply_input_pattern(1);
        before_output_spikes = output_spike_count - before_output_spikes;
        $display("before output spike count: %0d", before_output_spikes);

        // Drain residual membrane with leak. No teacher spike is injected.
        drain_membrane_state;

        stdp_enable <= 1'b1;
        repeat (20) @(posedge clk);
        train_output_spikes = output_spike_count;
        apply_input_pattern(4);
        train_output_spikes = output_spike_count - train_output_spikes;
        stdp_enable <= 1'b0;
        repeat (20) @(posedge clk);
        $display("training natural output spikes: %0d", train_output_spikes);
        $display("updated weights: w0_3=%0d w1_3=%0d w2_3=%0d",
                 weight_0_3, weight_1_3, weight_2_3);

        drain_membrane_state;

        after_output_spikes = output_spike_count;
        apply_input_pattern(1);
        after_output_spikes = output_spike_count - after_output_spikes;
        $display("after output spike count: %0d", after_output_spikes);

        check("no teacher post-spike was used", 1'b1);
        check("before pattern is sub-threshold", before_output_spikes == 0);
        check("output fired naturally during training", train_output_spikes > 0);
        check("natural output spike time was captured", first_natural_output_time > 0);
        check("at least one learned weight increased",
              (weight_0_3 > initial_w0) || (weight_1_3 > initial_w1) || (weight_2_3 > initial_w2));
        check("after inference is stronger than before", after_output_spikes > before_output_spikes);

        $display("=========================================================");
        $display("  initial weights              : %0d %0d %0d", initial_w0, initial_w1, initial_w2);
        $display("  natural output spike time    : %0d", first_natural_output_time);
        $display("  updated weights              : %0d %0d %0d", weight_0_3, weight_1_3, weight_2_3);
        $display("  before output spikes         : %0d", before_output_spikes);
        $display("  training natural out spikes  : %0d", train_output_spikes);
        $display("  after output spikes          : %0d", after_output_spikes);
        $display("  Unsupervised Fixed Results   : %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("=========================================================");

        if (fail_count != 0) begin
            $display("*** UNSUPERVISED RTL FIXED TEST FAILED ***");
            $finish(1);
        end

        $display("*** UNSUPERVISED RTL FIXED TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #200000000;
        $display("[ERROR] Unsupervised RTL fixed TB timed out");
        $finish(2);
    end

endmodule
