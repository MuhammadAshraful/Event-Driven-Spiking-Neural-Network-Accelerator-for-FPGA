//=============================================================================
// File-driven MNIST smoke test for the RTL-only unsupervised STDP path
//
// The spike file format is:
//   cycle neuron_id weight
//
// Labels are deliberately not read here. Training is unsupervised: natural input
// spikes must make output neuron 3 fire, and only that natural post spike can
// trigger an STDP update.
//=============================================================================

`timescale 1ns / 1ps

module tb_custom_rtl_unsupervised_mnist;

    localparam NUM_GROUPS        = 2;
    localparam WEIGHT_WIDTH      = 8;
    localparam GROUP_ID_WIDTH    = 1;
    localparam LOCAL_ID_WIDTH    = 4;
    localparam GLOBAL_ID_WIDTH   = 5;
    localparam MAX_EVENTS        = 4096;

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

    integer event_cycle [0:MAX_EVENTS-1];
    integer event_neuron [0:MAX_EVENTS-1];
    integer event_weight [0:MAX_EVENTS-1];
    integer event_count;
    integer pass_count;
    integer fail_count;
    integer before_output_spikes;
    integer train_output_spikes;
    integer inference_output_spikes;
    integer train_update_count;
    integer initial_w0;
    integer initial_w1;
    integer initial_w2;
    string spike_file;

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

    task automatic load_spike_file;
        integer fh;
        integer code;
        integer scratch;
    begin
        if (!$value$plusargs("SPIKE_FILE=%s", spike_file))
            spike_file = "mnist_selected.mem";

        fh = $fopen(spike_file, "r");
        if (fh == 0) begin
            $display("[ERROR] Could not open spike file: %0s", spike_file);
            $finish(2);
        end

        event_count = 0;
        while (!$feof(fh) && event_count < MAX_EVENTS) begin
            code = $fscanf(fh, "%d %d %d\n",
                           event_cycle[event_count],
                           event_neuron[event_count],
                           event_weight[event_count]);
            if (code == 3) begin
                event_count = event_count + 1;
            end else begin
                scratch = $fgetc(fh);
            end
        end
        $fclose(fh);
        $display("loaded spike file: %0s events=%0d", spike_file, event_count);
    end
    endtask

    task automatic replay_events;
        integer idx;
        integer last_cycle;
        integer gap;
        integer clipped_neuron;
        integer clipped_weight;
    begin
        last_cycle = 0;
        for (idx = 0; idx < event_count; idx = idx + 1) begin
            gap = event_cycle[idx] - last_cycle;
            if (gap > 0)
                repeat (gap) @(posedge clk);
            last_cycle = event_cycle[idx];

            clipped_neuron = event_neuron[idx] & 15;
            clipped_weight = event_weight[idx] & 255;
            inject_spike(global_id(1'd0, clipped_neuron[LOCAL_ID_WIDTH-1:0]),
                         clipped_weight[WEIGHT_WIDTH-1:0]);
        end
        wait_all_idle;
    end
    endtask

    always @(posedge clk) begin
        if (natural_output_observed) begin
            $display("mnist natural output spike: neuron=3 time=%0d stdp_enable=%0d",
                     current_time_out, stdp_enable);
        end
        if (learned_update_pulse) begin
            $display("mnist STDP update: src=%0d dst=3 weight=%0d time=%0d",
                     learned_update_src, learned_update_weight, current_time_out);
        end
    end

    initial begin
        $display("=========================================================");
        $display("  MNIST file-driven unsupervised RTL STDP smoke TB");
        $display("=========================================================");

        pass_count = 0;
        fail_count = 0;
        before_output_spikes = 0;
        train_output_spikes = 0;
        inference_output_spikes = 0;
        train_update_count = 0;

        load_spike_file;
        do_reset;

        program_weight(4'd0, 4'd3, 8'd4);
        program_weight(4'd1, 4'd3, 8'd4);
        program_weight(4'd2, 4'd3, 8'd4);
        initial_w0 = weight_0_3;
        initial_w1 = weight_1_3;
        initial_w2 = weight_2_3;

        stdp_enable <= 1'b0;
        repeat (20) @(posedge clk);
        before_output_spikes = output_spike_count;
        replay_events;
        before_output_spikes = output_spike_count - before_output_spikes;

        drain_membrane_state;

        stdp_enable <= 1'b1;
        repeat (20) @(posedge clk);
        train_output_spikes = output_spike_count;
        train_update_count = learned_update_count;
        replay_events;
        train_output_spikes = output_spike_count - train_output_spikes;
        train_update_count = learned_update_count - train_update_count;
        stdp_enable <= 1'b0;
        repeat (20) @(posedge clk);

        drain_membrane_state;

        inference_output_spikes = output_spike_count;
        replay_events;
        inference_output_spikes = output_spike_count - inference_output_spikes;

        $display("MNIST RTL Summary: file=%0s input_spikes=%0d before_spikes=%0d train_spikes=%0d inference_spikes=%0d learned_updates=%0d weights=%0d,%0d,%0d winner=3",
                 spike_file, event_count, before_output_spikes, train_output_spikes,
                 inference_output_spikes, train_update_count, weight_0_3, weight_1_3, weight_2_3);

        check("MNIST spike file contains events", event_count > 0);
        check("training uses natural output spikes", train_output_spikes > 0);
        check("STDP learned at least one weight", train_update_count > 0);
        check("learned weights increased",
              (weight_0_3 > initial_w0) || (weight_1_3 > initial_w1) || (weight_2_3 > initial_w2));
        check("inference runs with learning disabled", stdp_enable == 1'b0);
        check("inference produces output spikes", inference_output_spikes > 0);
        check("after-learning inference is not weaker", inference_output_spikes >= before_output_spikes);

        $display("MNIST RTL Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        if (fail_count != 0) begin
            $display("*** MNIST UNSUPERVISED RTL SMOKE TEST FAILED ***");
            $finish(1);
        end

        $display("*** MNIST UNSUPERVISED RTL SMOKE TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #400000000;
        $display("[ERROR] MNIST unsupervised RTL TB timed out");
        $finish(2);
    end

endmodule
