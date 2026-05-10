//=============================================================================
// Unit test for stdp_learning_engine.v
//=============================================================================

`timescale 1ns / 1ps

module tb_stdp_learning_engine;

    localparam NUM_NEURONS     = 4;
    localparam NEURON_ID_WIDTH = 2;
    localparam WEIGHT_WIDTH    = 8;
    localparam TIME_WIDTH      = 16;
    localparam STDP_WINDOW     = 10;
    localparam A_PLUS          = 5;
    localparam A_MINUS         = 3;
    localparam W_MIN           = 0;
    localparam W_MAX           = 15;

    localparam RULE_NONE = 2'd0;
    localparam RULE_LTP  = 2'd1;
    localparam RULE_LTD  = 2'd2;

    reg clk;
    reg rst_n;
    reg enable;
    reg [TIME_WIDTH-1:0] current_time;
    reg pre_spike_valid;
    reg [NEURON_ID_WIDTH-1:0] pre_neuron_id;
    reg post_spike_valid;
    reg [NEURON_ID_WIDTH-1:0] post_neuron_id;
    reg [WEIGHT_WIDTH-1:0] current_weight;
    reg update_ready;

    wire update_valid;
    wire [NEURON_ID_WIDTH-1:0] update_src;
    wire [NEURON_ID_WIDTH-1:0] update_dst;
    wire [WEIGHT_WIDTH-1:0] update_weight;
    wire update_exc;
    wire signed [WEIGHT_WIDTH:0] debug_delta;
    wire [1:0] debug_rule_applied;

    integer pass_count;
    integer fail_count;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    stdp_learning_engine #(
        .NUM_NEURONS(NUM_NEURONS),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .TIME_WIDTH(TIME_WIDTH),
        .STDP_WINDOW(STDP_WINDOW),
        .A_PLUS(A_PLUS),
        .A_MINUS(A_MINUS),
        .W_MIN(W_MIN),
        .W_MAX(W_MAX)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .current_time(current_time),
        .pre_spike_valid(pre_spike_valid),
        .pre_neuron_id(pre_neuron_id),
        .post_spike_valid(post_spike_valid),
        .post_neuron_id(post_neuron_id),
        .current_weight(current_weight),
        .update_ready(update_ready),
        .update_valid(update_valid),
        .update_src(update_src),
        .update_dst(update_dst),
        .update_weight(update_weight),
        .update_exc(update_exc),
        .debug_delta(debug_delta),
        .debug_rule_applied(debug_rule_applied)
    );

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
        current_time <= 0;
        pre_spike_valid <= 1'b0;
        pre_neuron_id <= 0;
        post_spike_valid <= 1'b0;
        post_neuron_id <= 0;
        current_weight <= 0;
        update_ready <= 1'b1;
        repeat (5) @(posedge clk);
        rst_n <= 1'b1;
        enable <= 1'b1;
        repeat (2) @(posedge clk);
    end
    endtask

    task automatic pulse_pre;
        input [NEURON_ID_WIDTH-1:0] nid;
        input [TIME_WIDTH-1:0] t;
    begin
        @(posedge clk);
        current_time <= t;
        pre_neuron_id <= nid;
        pre_spike_valid <= 1'b1;
        @(posedge clk);
        pre_spike_valid <= 1'b0;
    end
    endtask

    task automatic pulse_post;
        input [NEURON_ID_WIDTH-1:0] nid;
        input [TIME_WIDTH-1:0] t;
    begin
        @(posedge clk);
        current_time <= t;
        post_neuron_id <= nid;
        post_spike_valid <= 1'b1;
        @(posedge clk);
        post_spike_valid <= 1'b0;
    end
    endtask

    task automatic clear_update;
    begin
        @(posedge clk);
        #1;
    end
    endtask

    initial begin
        $display("=========================================================");
        $display("  STDP learning engine unit test");
        $display("=========================================================");

        pass_count = 0;
        fail_count = 0;

        do_reset;
        check("reset clears update_valid", update_valid == 1'b0);

        // LTP: pre fires before post, so weight increases.
        do_reset;
        current_weight <= 8'd5;
        pulse_pre(2'd0, 16'd10);
        pulse_post(2'd1, 16'd14);
        #1;
        check("LTP produces an update", update_valid);
        check("LTP source/destination are pre0->post1", update_src == 0 && update_dst == 1);
        check("LTP increases weight by A_PLUS", update_weight == 8'd10);
        check("LTP debug rule is set", debug_rule_applied == RULE_LTP && debug_delta == 5);
        clear_update;

        // LTD: post fires before pre, so weight decreases.
        do_reset;
        current_weight <= 8'd8;
        pulse_post(2'd1, 16'd20);
        pulse_pre(2'd0, 16'd24);
        #1;
        check("LTD produces an update", update_valid);
        check("LTD source/destination are pre0->post1", update_src == 0 && update_dst == 1);
        check("LTD decreases weight by A_MINUS", update_weight == 8'd5);
        check("LTD debug rule is set", debug_rule_applied == RULE_LTD && debug_delta == -3);
        clear_update;

        // Outside the window: no update.
        do_reset;
        current_weight <= 8'd5;
        pulse_pre(2'd0, 16'd1);
        pulse_post(2'd1, 16'd30);
        #1;
        check("outside window produces no update", update_valid == 1'b0);

        // Saturation at W_MAX.
        do_reset;
        current_weight <= 8'd13;
        pulse_pre(2'd0, 16'd3);
        pulse_post(2'd1, 16'd4);
        #1;
        check("LTP saturates at W_MAX", update_valid && update_weight == W_MAX);
        check("LTP saturation reports actual applied delta", debug_delta == 2);
        clear_update;

        // Saturation at W_MIN.
        do_reset;
        current_weight <= 8'd2;
        pulse_post(2'd1, 16'd3);
        pulse_pre(2'd0, 16'd4);
        #1;
        check("LTD saturates at W_MIN", update_valid && update_weight == W_MIN);
        check("LTD saturation reports actual applied delta", debug_delta == -2);
        clear_update;

        $display("=========================================================");
        $display("  STDP Unit Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("=========================================================");

        if (fail_count != 0) begin
            $display("*** STDP LEARNING ENGINE TEST FAILED ***");
            $finish(1);
        end

        $display("*** STDP LEARNING ENGINE TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #2000000;
        $display("[ERROR] STDP unit test timed out");
        $finish(2);
    end

endmodule
