//=============================================================================
// Integrated RTL-only STDP learning test
//=============================================================================

`timescale 1ns / 1ps

module tb_custom_rtl_stdp_learning;

    localparam NUM_GROUPS        = 2;
    localparam NEURONS_PER_GROUP = 16;
    localparam WEIGHT_WIDTH      = 8;
    localparam GROUP_ID_WIDTH    = 1;
    localparam LOCAL_ID_WIDTH    = 4;
    localparam GLOBAL_ID_WIDTH   = 5;

    localparam RULE_LTP = 2'd1;

    reg clk;
    reg rst_n;
    reg enable;
    reg stdp_enable;

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
    wire [15:0]                 g1_spike_count;
    wire                        router_busy;
    wire [NUM_GROUPS-1:0]       group_busy;
    wire [15:0]                 current_time_out;
    wire [WEIGHT_WIDTH-1:0]     tracked_weight_0_1;
    wire                        learned_update_pulse;
    wire                        stdp_update_valid;
    wire [LOCAL_ID_WIDTH-1:0]   stdp_update_src;
    wire [LOCAL_ID_WIDTH-1:0]   stdp_update_dst;
    wire [WEIGHT_WIDTH-1:0]     stdp_update_weight;
    wire signed [WEIGHT_WIDTH:0] stdp_debug_delta;
    wire [1:0]                  stdp_debug_rule_applied;
    wire                        stdp_pre_observed;
    wire                        stdp_post_observed;

    integer pass_count;
    integer fail_count;
    integer initial_weight;
    integer updated_weight;
    integer before_spikes;
    integer after_spikes;
    integer pre_spike_time;
    integer post_spike_time;
    integer ltp_seen;
    integer last_rule;
    integer last_delta;

    custom_rtl_stdp_sim_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .stdp_enable(stdp_enable),
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
        .g1_spike_count(g1_spike_count),
        .router_busy(router_busy),
        .group_busy(group_busy),
        .current_time_out(current_time_out),
        .tracked_weight_0_1(tracked_weight_0_1),
        .learned_update_pulse(learned_update_pulse),
        .stdp_update_valid(stdp_update_valid),
        .stdp_update_src(stdp_update_src),
        .stdp_update_dst(stdp_update_dst),
        .stdp_update_weight(stdp_update_weight),
        .stdp_debug_delta(stdp_debug_delta),
        .stdp_debug_rule_applied(stdp_debug_rule_applied),
        .stdp_pre_observed(stdp_pre_observed),
        .stdp_post_observed(stdp_post_observed)
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
        ext_spike_valid <= 1'b0;
        ext_spike_neuron_id <= {GLOBAL_ID_WIDTH{1'b0}};
        ext_spike_weight <= {WEIGHT_WIDTH{1'b0}};
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
        while ((router_busy || group_busy != 0) && timeout < 50000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        repeat (100) @(posedge clk);
    end
    endtask

    task automatic program_weight_0_1;
        input [WEIGHT_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        host_weight_we <= 2'b01;
        host_weight_src <= 4'd0;
        host_weight_dst <= 4'd1;
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
        while (!ext_spike_ready && timeout < 5000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        // event_router_ng first sees ext_spike_valid in ST_IDLE, then uses the
        // still-held data on the following ST_EXT_ROUTE cycle.
        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        ext_spike_valid <= 1'b0;
        ext_spike_weight <= 8'd0;
        ext_spike_neuron_id <= {GLOBAL_ID_WIDTH{1'b0}};
        @(posedge clk);
    end
    endtask

    task automatic wait_for_ltp_update;
        integer timeout;
    begin
        timeout = 0;
        while (!ltp_seen && timeout < 100000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (ltp_seen) begin
            updated_weight = tracked_weight_0_1;
        end
    end
    endtask

    always @(posedge clk) begin
        if (stdp_pre_observed) begin
            $display("Router learn pre observation: neuron=0 time=%0d stdp_enable=%0d",
                     current_time_out, stdp_enable);
        end
        if (stdp_post_observed) begin
            $display("Router learn post observation: neuron=1 time=%0d stdp_enable=%0d",
                     current_time_out, stdp_enable);
        end
        if (stdp_enable && stdp_pre_observed) begin
            pre_spike_time = current_time_out;
            $display("STDP observed pre spike: neuron=0 time=%0d", current_time_out);
        end
        if (stdp_enable && stdp_post_observed) begin
            post_spike_time = current_time_out;
            $display("STDP observed post spike: neuron=1 time=%0d", current_time_out);
        end
        if (learned_update_pulse) begin
            ltp_seen = 1;
            updated_weight = tracked_weight_0_1;
            last_rule = stdp_debug_rule_applied;
            last_delta = stdp_debug_delta;
            $display("STDP update applied: src=%0d dst=%0d weight=%0d delta=%0d rule=%0d time=%0d",
                     stdp_update_src, stdp_update_dst, tracked_weight_0_1,
                     stdp_debug_delta, stdp_debug_rule_applied, current_time_out);
        end
    end

    initial begin
        $display("=========================================================");
        $display("  Custom RTL-only STDP learning integration TB");
        $display("=========================================================");

        pass_count = 0;
        fail_count = 0;
        initial_weight = 0;
        updated_weight = 0;
        before_spikes = 0;
        after_spikes = 0;
        pre_spike_time = 0;
        post_spike_time = 0;
        ltp_seen = 0;
        last_rule = 0;
        last_delta = 0;

        // Before-learning run.
        do_reset;
        program_weight_0_1(8'd5);
        initial_weight = tracked_weight_0_1;
        $display("initial weight n0->n1: %0d", initial_weight);

        before_spikes = g0_spike_count;
        inject_spike(global_id(1'd0, 4'd0), 8'd12);
        wait_all_idle;
        before_spikes = g0_spike_count - before_spikes;
        $display("output spikes before learning: %0d", before_spikes);
        $display("group0 total after before run: %0d", g0_spike_count);

        // The core_group RAMs are initialized at simulation start, but this RTL
        // does not clear neuron state RAM on reset. The sub-threshold before
        // run leaves neuron1 membrane at 5, so force neuron1 to fire with
        // learning disabled to reset its membrane before the learning trial.
        $display("clear teacher neuron state: neuron 1, weight 12");
        inject_spike(global_id(1'd0, 4'd1), 8'd12);
        wait_all_idle;
        repeat (1000) @(posedge clk);
        $display("group0 total after clear     : %0d", g0_spike_count);

        stdp_enable <= 1'b1;
        repeat (20) @(posedge clk);

        // Learning run: pre0 fires, then a teacher/forced post1 spike fires.
        $display("learning pre input       : neuron 0, weight 12");
        inject_spike(global_id(1'd0, 4'd0), 8'd12);
        repeat (200) @(posedge clk);
        $display("group0 total after learn pre : %0d", g0_spike_count);

        $display("teacher post input       : neuron 1, weight 12");
        inject_spike(global_id(1'd0, 4'd1), 8'd12);
        wait_for_ltp_update;
        wait_all_idle;
        updated_weight = tracked_weight_0_1;
        stdp_enable <= 1'b0;
        repeat (20) @(posedge clk);
        $display("group0 total after teacher   : %0d", g0_spike_count);

        // After-learning run.
        repeat (1000) @(posedge clk);
        after_spikes = g0_spike_count;
        inject_spike(global_id(1'd0, 4'd0), 8'd12);
        wait_all_idle;
        after_spikes = g0_spike_count - after_spikes;
        $display("output spikes after learning : %0d", after_spikes);

        check("initial weight is sub-threshold", initial_weight == 5);
        check("pre spike was observed by STDP", pre_spike_time > 0);
        check("post/teacher spike was observed by STDP", post_spike_time > pre_spike_time);
        check("LTP rule was applied", ltp_seen && last_rule == RULE_LTP && last_delta > 0);
        check("weight increased to threshold", updated_weight >= 10);
        check("before-learning run fired only source neuron", before_spikes == 1);
        check("after-learning run fired source and target neuron", after_spikes >= 2);

        $display("=========================================================");
        $display("  initial weight             : %0d", initial_weight);
        $display("  pre spike time             : %0d", pre_spike_time);
        $display("  post/teacher spike time    : %0d", post_spike_time);
        $display("  STDP rule applied          : %0d", last_rule);
        $display("  updated weight             : %0d", updated_weight);
        $display("  output spikes before/after : %0d / %0d", before_spikes, after_spikes);
        $display("  Custom RTL STDP Results    : %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("=========================================================");

        if (fail_count != 0) begin
            $display("*** CUSTOM RTL STDP LEARNING TEST FAILED ***");
            $finish(1);
        end

        $display("*** CUSTOM RTL STDP LEARNING TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #150000000;
        $display("[ERROR] Custom RTL STDP learning TB timed out");
        $finish(2);
    end

endmodule
