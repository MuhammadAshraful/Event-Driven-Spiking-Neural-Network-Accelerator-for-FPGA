//=============================================================================
// Testbench: generated HLS learning engine + custom RTL integration top
//
// Experiment:
//   1. Initial intra-group weight n0->n1 is sub-threshold.
//   2. HLS receives a pre spike and drives RTL; n0 fires, n1 does not.
//   3. A teacher post spike is supplied to HLS.
//   4. HLS checkpoint output is adapted into event_router_ng.learn_weight_*.
//   5. The same pre input is applied again; n1 now fires.
//=============================================================================

`timescale 1ns / 1ps

module tb_custom_hls_rtl_learning;

    localparam ADDR_AP_CTRL       = 8'h00;
    localparam ADDR_CTRL_REG      = 8'h10;
    localparam ADDR_CONFIG_REG    = 8'h18;
    localparam ADDR_MODE_REG      = 8'h20;
    localparam ADDR_TIME_STEPS    = 8'h28;

    localparam CTRL_ENABLE        = 32'h0000_0001;
    localparam CTRL_LEARNING      = 32'h0000_0008;
    localparam CTRL_WEIGHT_READ   = 32'h0000_0010;

    localparam MODE_INFERENCE     = 32'd0;
    localparam MODE_TRAIN_STDP    = 32'd1;
    localparam MODE_CHECKPOINT    = 32'd2;

    reg clk;
    reg rst_n;
    reg rtl_enable;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg  [31:0] s_axis_spikes_TDATA;
    reg         s_axis_spikes_TVALID;
    wire        s_axis_spikes_TREADY;
    reg  [0:0]  s_axis_spikes_TLAST;

    reg         teacher_post_valid;
    reg  [7:0]  teacher_post_neuron_id;
    reg  [7:0]  teacher_post_weight;

    reg  [1:0]  host_weight_we;
    reg  [3:0]  host_weight_src;
    reg  [3:0]  host_weight_dst;
    reg  [7:0]  host_weight_data;
    reg         host_weight_exc;

    reg         s_axi_ctrl_AWVALID;
    wire        s_axi_ctrl_AWREADY;
    reg  [7:0]  s_axi_ctrl_AWADDR;
    reg         s_axi_ctrl_WVALID;
    wire        s_axi_ctrl_WREADY;
    reg  [31:0] s_axi_ctrl_WDATA;
    reg  [3:0]  s_axi_ctrl_WSTRB;
    reg         s_axi_ctrl_ARVALID;
    wire        s_axi_ctrl_ARREADY;
    reg  [7:0]  s_axi_ctrl_ARADDR;
    wire        s_axi_ctrl_RVALID;
    reg         s_axi_ctrl_RREADY;
    wire [31:0] s_axi_ctrl_RDATA;
    wire [1:0]  s_axi_ctrl_RRESP;
    wire        s_axi_ctrl_BVALID;
    reg         s_axi_ctrl_BREADY;
    wire [1:0]  s_axi_ctrl_BRESP;
    wire        interrupt;

    wire [15:0] g0_spike_count;
    wire [15:0] g1_spike_count;
    wire        router_busy;
    wire [1:0]  group_busy;
    wire [7:0]  applied_weight_0_1;
    wire        learned_update_valid;
    wire [7:0]  learned_update_weight;
    wire [31:0] hls_weight_stream_data;
    wire        hls_weight_stream_valid;

    integer fail_count;
    integer pass_count;
    integer before_spikes;
    integer after_spikes;
    integer initial_weight;
    integer learned_weight;
    integer update_seen;

    custom_hls_rtl_sim_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .rtl_enable(rtl_enable),
        .s_axis_spikes_TDATA(s_axis_spikes_TDATA),
        .s_axis_spikes_TVALID(s_axis_spikes_TVALID),
        .s_axis_spikes_TREADY(s_axis_spikes_TREADY),
        .s_axis_spikes_TLAST(s_axis_spikes_TLAST),
        .teacher_post_valid(teacher_post_valid),
        .teacher_post_neuron_id(teacher_post_neuron_id),
        .teacher_post_weight(teacher_post_weight),
        .host_weight_we(host_weight_we),
        .host_weight_src(host_weight_src),
        .host_weight_dst(host_weight_dst),
        .host_weight_data(host_weight_data),
        .host_weight_exc(host_weight_exc),
        .s_axi_ctrl_AWVALID(s_axi_ctrl_AWVALID),
        .s_axi_ctrl_AWREADY(s_axi_ctrl_AWREADY),
        .s_axi_ctrl_AWADDR(s_axi_ctrl_AWADDR),
        .s_axi_ctrl_WVALID(s_axi_ctrl_WVALID),
        .s_axi_ctrl_WREADY(s_axi_ctrl_WREADY),
        .s_axi_ctrl_WDATA(s_axi_ctrl_WDATA),
        .s_axi_ctrl_WSTRB(s_axi_ctrl_WSTRB),
        .s_axi_ctrl_ARVALID(s_axi_ctrl_ARVALID),
        .s_axi_ctrl_ARREADY(s_axi_ctrl_ARREADY),
        .s_axi_ctrl_ARADDR(s_axi_ctrl_ARADDR),
        .s_axi_ctrl_RVALID(s_axi_ctrl_RVALID),
        .s_axi_ctrl_RREADY(s_axi_ctrl_RREADY),
        .s_axi_ctrl_RDATA(s_axi_ctrl_RDATA),
        .s_axi_ctrl_RRESP(s_axi_ctrl_RRESP),
        .s_axi_ctrl_BVALID(s_axi_ctrl_BVALID),
        .s_axi_ctrl_BREADY(s_axi_ctrl_BREADY),
        .s_axi_ctrl_BRESP(s_axi_ctrl_BRESP),
        .interrupt(interrupt),
        .g0_spike_count(g0_spike_count),
        .g1_spike_count(g1_spike_count),
        .router_busy(router_busy),
        .group_busy(group_busy),
        .applied_weight_0_1(applied_weight_0_1),
        .learned_update_valid(learned_update_valid),
        .learned_update_weight(learned_update_weight),
        .hls_weight_stream_data(hls_weight_stream_data),
        .hls_weight_stream_valid(hls_weight_stream_valid)
    );

    function [31:0] spike_word;
        input [7:0] neuron_id;
        input [7:0] weight;
    begin
        spike_word = {14'd0, weight, 2'b00, neuron_id};
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

    task automatic axi_write;
        input [7:0] addr;
        input [31:0] data;
        integer timeout;
    begin
        @(posedge clk);
        s_axi_ctrl_AWADDR  <= addr;
        s_axi_ctrl_WDATA   <= data;
        s_axi_ctrl_WSTRB   <= 4'hF;
        s_axi_ctrl_AWVALID <= 1'b1;
        s_axi_ctrl_WVALID  <= 1'b1;
        s_axi_ctrl_BREADY  <= 1'b1;

        timeout = 0;
        while (!s_axi_ctrl_AWREADY && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        @(posedge clk);
        s_axi_ctrl_AWVALID <= 1'b0;

        timeout = 0;
        while (!s_axi_ctrl_WREADY && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        @(posedge clk);
        s_axi_ctrl_WVALID <= 1'b0;

        timeout = 0;
        while (!s_axi_ctrl_BVALID && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        @(posedge clk);
        s_axi_ctrl_BREADY <= 1'b0;
    end
    endtask

    task automatic axi_read;
        input [7:0] addr;
        output [31:0] data;
        integer timeout;
    begin
        @(posedge clk);
        s_axi_ctrl_ARADDR  <= addr;
        s_axi_ctrl_ARVALID <= 1'b1;
        s_axi_ctrl_RREADY  <= 1'b1;

        timeout = 0;
        while (!s_axi_ctrl_ARREADY && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        @(posedge clk);
        s_axi_ctrl_ARVALID <= 1'b0;

        timeout = 0;
        while (!s_axi_ctrl_RVALID && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        data = s_axi_ctrl_RDATA;
        @(posedge clk);
        s_axi_ctrl_RREADY <= 1'b0;
    end
    endtask

    task automatic wait_ap_done;
        input integer max_polls;
        output integer done_seen;
        reg [31:0] ctrl_status;
        integer polls;
    begin
        done_seen = 0;
        polls = 0;
        while (!done_seen && polls < max_polls) begin
            axi_read(ADDR_AP_CTRL, ctrl_status);
            done_seen = ctrl_status[1];
            polls = polls + 1;
        end
        if (!done_seen)
            $display("[ERROR] wait_ap_done timed out after %0d polls", max_polls);
    end
    endtask

    task automatic start_kernel;
    begin
        axi_write(ADDR_AP_CTRL, 32'h0000_0001);
    end
    endtask

    task automatic wait_all_idle;
        integer timeout;
    begin
        timeout = 0;
        while ((router_busy || group_busy != 0) && timeout < 20000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        repeat (50) @(posedge clk);
    end
    endtask

    task automatic program_initial_weight;
        input [7:0] weight;
    begin
        @(posedge clk);
        host_weight_we   <= 2'b01;
        host_weight_src  <= 4'd0;
        host_weight_dst  <= 4'd1;
        host_weight_data <= weight;
        host_weight_exc  <= 1'b1;
        @(posedge clk);
        host_weight_we <= 2'b00;
        repeat (4) @(posedge clk);
    end
    endtask

    task automatic run_hls_pre_to_rtl;
        input integer learning_enabled;
        integer done_seen;
        integer timeout;
    begin
        axi_write(ADDR_MODE_REG, learning_enabled ? MODE_TRAIN_STDP : MODE_INFERENCE);
        axi_write(ADDR_TIME_STEPS, 32'd1);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE | (learning_enabled ? CTRL_LEARNING : 32'd0));
        s_axis_spikes_TDATA  <= spike_word(8'd0, 8'd12);
        s_axis_spikes_TVALID <= 1'b1;
        s_axis_spikes_TLAST  <= 1'b1;
        start_kernel;

        timeout = 0;
        while (!(s_axis_spikes_TVALID && s_axis_spikes_TREADY) && timeout < 300000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 300000)
            $display("[ERROR] HLS pre spike was not accepted");
        @(posedge clk);
        s_axis_spikes_TVALID <= 1'b0;

        wait_ap_done(200000, done_seen);
        check("HLS pre-spike kernel completed", done_seen);
        wait_all_idle;
    end
    endtask

    task automatic run_teacher_post;
        integer done_seen;
    begin
        axi_write(ADDR_MODE_REG, MODE_TRAIN_STDP);
        axi_write(ADDR_TIME_STEPS, 32'd1);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE | CTRL_LEARNING);
        teacher_post_neuron_id <= 8'd1;
        teacher_post_weight    <= 8'd0;
        teacher_post_valid     <= 1'b1;
        start_kernel;
        wait_ap_done(200000, done_seen);
        teacher_post_valid <= 1'b0;
        check("HLS teacher post-spike kernel completed", done_seen);
    end
    endtask

    task automatic run_checkpoint_apply;
        integer done_seen;
        integer timeout;
    begin
        update_seen = 0;
        axi_write(ADDR_MODE_REG, MODE_CHECKPOINT);
        axi_write(ADDR_TIME_STEPS, 32'd4);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE | CTRL_WEIGHT_READ);
        start_kernel;

        timeout = 0;
        while (!update_seen && timeout < 300000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        wait_ap_done(200000, done_seen);
        check("HLS checkpoint/apply kernel completed", done_seen);
        check("HLS checkpoint adapter observed learned weight", update_seen);
        wait_all_idle;
    end
    endtask

    always @(posedge clk) begin
        if (hls_weight_stream_valid) begin
            $display("HLS weight stream: data=0x%08h column=%0d weight=%0d time=%0t",
                     hls_weight_stream_data,
                     hls_weight_stream_data[15:8],
                     hls_weight_stream_data[23:16],
                     $time);
        end
        if (learned_update_valid) begin
            update_seen = 1;
            $display("HLS->RTL learned update adapter: G0 n0->n1 weight=%0d time=%0t",
                     learned_update_weight, $time);
        end
    end

    initial begin
        $display("=========================================================");
        $display("  Custom HLS+RTL learning integration TB");
        $display("=========================================================");

        fail_count = 0;
        pass_count = 0;
        before_spikes = 0;
        after_spikes = 0;
        initial_weight = 0;
        learned_weight = 0;
        update_seen = 0;

        rst_n = 1'b0;
        rtl_enable = 1'b0;
        s_axis_spikes_TDATA = 32'd0;
        s_axis_spikes_TVALID = 1'b0;
        s_axis_spikes_TLAST = 1'b0;
        teacher_post_valid = 1'b0;
        teacher_post_neuron_id = 8'd0;
        teacher_post_weight = 8'd0;
        host_weight_we = 2'b00;
        host_weight_src = 4'd0;
        host_weight_dst = 4'd0;
        host_weight_data = 8'd0;
        host_weight_exc = 1'b1;
        s_axi_ctrl_AWVALID = 1'b0;
        s_axi_ctrl_AWADDR = 8'd0;
        s_axi_ctrl_WVALID = 1'b0;
        s_axi_ctrl_WDATA = 32'd0;
        s_axi_ctrl_WSTRB = 4'hF;
        s_axi_ctrl_ARVALID = 1'b0;
        s_axi_ctrl_ARADDR = 8'd0;
        s_axi_ctrl_RREADY = 1'b0;
        s_axi_ctrl_BREADY = 1'b0;

        repeat (20) @(posedge clk);
        rst_n <= 1'b1;
        rtl_enable <= 1'b1;
        repeat (50) @(posedge clk);

        axi_write(ADDR_CONFIG_REG, {16'd0, 16'd10});

        program_initial_weight(8'd5);
        initial_weight = applied_weight_0_1;

        $display("initial weight n0->n1 : %0d", initial_weight);
        $display("pre spike             : neuron 0, weight 12");

        before_spikes = g0_spike_count;
        run_hls_pre_to_rtl(1);
        before_spikes = g0_spike_count - before_spikes;
        $display("output spike count before learning: %0d", before_spikes);

        $display("teacher post spike    : neuron 1");
        run_teacher_post;

        run_checkpoint_apply;
        repeat (20) @(posedge clk);
        learned_weight = applied_weight_0_1;
        $display("HLS learned/applied RTL weight n0->n1: %0d", learned_weight);

        repeat (500) @(posedge clk);
        after_spikes = g0_spike_count;
        run_hls_pre_to_rtl(0);
        after_spikes = g0_spike_count - after_spikes;
        $display("output spike count after learning : %0d", after_spikes);

        check("initial weight is sub-threshold", initial_weight < 10);
        check("HLS learned weight is visible and stronger", learned_weight > initial_weight);
        check("before-learning run fired only source neuron", before_spikes == 1);
        check("after-learning run fired source and learned target", after_spikes >= 2);

        $display("=========================================================");
        $display("  pre neuron                : 0");
        $display("  post neuron               : 1");
        $display("  initial weight            : %0d", initial_weight);
        $display("  HLS weight update/result  : %0d", learned_weight);
        $display("  applied RTL weight        : %0d", applied_weight_0_1);
        $display("  output spikes before/after: %0d / %0d", before_spikes, after_spikes);
        $display("  Custom Integration Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("=========================================================");

        if (fail_count != 0) begin
            $display("*** CUSTOM HLS RTL LEARNING TEST FAILED ***");
            $finish(1);
        end

        $display("*** CUSTOM HLS RTL LEARNING TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #150000000;
        $display("[ERROR] Custom HLS RTL learning TB timed out");
        $finish(2);
    end

endmodule
