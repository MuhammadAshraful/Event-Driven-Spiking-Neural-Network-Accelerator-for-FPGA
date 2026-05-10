//=============================================================================
// Testbench: one-image MNIST-style spike-file simulation through generated HLS
// + custom RTL integration top.
//
// The spike file format is:
//   logical_cycle neuron_id weight
//
// This is a small smoke test, not a full MNIST workload. The provided file is
// a 4x4 pooled event representation of one MNIST test image.
//=============================================================================

`timescale 1ns / 1ps

module tb_mnist_hls_rtl_sim;

    localparam ADDR_AP_CTRL       = 8'h00;
    localparam ADDR_CTRL_REG      = 8'h10;
    localparam ADDR_CONFIG_REG    = 8'h18;
    localparam ADDR_MODE_REG      = 8'h20;
    localparam ADDR_TIME_STEPS    = 8'h28;

    localparam CTRL_ENABLE        = 32'h0000_0001;
    localparam MODE_INFERENCE     = 32'd0;
    localparam MEM_FILE           = "mnist_one_image_spikes.mem";
    localparam EXPECTED_SPIKES    = 2;

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
    integer event_count;
    integer logical_cycle;
    integer neuron_id;
    integer weight;
    integer previous_cycle;
    integer spike_count_delta;
    integer fd;
    integer rc;

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
        input [7:0] nid;
        input [7:0] wgt;
    begin
        spike_word = {14'd0, wgt, 2'b00, nid};
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
        repeat (30) @(posedge clk);
    end
    endtask

    task automatic run_event;
        input [7:0] nid;
        input [7:0] wgt;
        input integer ev_cycle;
        integer done_seen;
        integer timeout;
    begin
        axi_write(ADDR_MODE_REG, MODE_INFERENCE);
        axi_write(ADDR_TIME_STEPS, 32'd1);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE);

        s_axis_spikes_TDATA  <= spike_word(nid, wgt);
        s_axis_spikes_TVALID <= 1'b1;
        s_axis_spikes_TLAST  <= 1'b1;
        start_kernel;

        timeout = 0;
        while (!(s_axis_spikes_TVALID && s_axis_spikes_TREADY) && timeout < 300000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 300000)
            $display("[ERROR] MNIST-style event was not accepted");
        @(posedge clk);
        s_axis_spikes_TVALID <= 1'b0;

        wait_ap_done(200000, done_seen);
        check("HLS accepted MNIST-style spike event", done_seen && timeout < 300000);
        wait_all_idle;
        $display("MNIST-style event: logical_cycle=%0d neuron=%0d weight=%0d g0_spikes=%0d",
                 ev_cycle, nid, wgt, g0_spike_count);
    end
    endtask

    always @(posedge clk) begin
        if (dut.grp_in_valid[0]) begin
            $display("RTL router delivery: group0 neuron=%0d weight=%0d exc=%0d time=%0t",
                     dut.grp_in_dest_id[3:0],
                     dut.grp_in_weight[7:0],
                     dut.grp_in_exc[0],
                     $time);
        end
        if (dut.grp_spike_valid[0] && dut.grp_spike_ready[0]) begin
            $display("RTL group0 output spike: neuron=%0d time=%0t",
                     dut.grp_spike_neuron_id[3:0],
                     $time);
        end
    end

    initial begin
        $display("=========================================================");
        $display("  MNIST-style HLS+RTL spike-file simulation TB");
        $display("=========================================================");
        $display("source image      : MNIST test image 0, 4x4 pooled");
        $display("spike file        : %0s", MEM_FILE);

        fail_count = 0;
        pass_count = 0;
        event_count = 0;
        previous_cycle = 0;
        spike_count_delta = 0;

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
        spike_count_delta = g0_spike_count;

        fd = $fopen(MEM_FILE, "r");
        if (fd == 0) begin
            $display("[FAIL] could not open %0s", MEM_FILE);
            $finish(1);
        end

        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%d %d %d\n", logical_cycle, neuron_id, weight);
            if (rc == 3) begin
                if (logical_cycle > previous_cycle)
                    repeat (logical_cycle - previous_cycle) @(posedge clk);
                previous_cycle = logical_cycle;
                event_count = event_count + 1;
                run_event(neuron_id[7:0], weight[7:0], logical_cycle);
            end
        end
        $fclose(fd);

        wait_all_idle;
        spike_count_delta = g0_spike_count - spike_count_delta;

        check("MNIST-style spike file had events", event_count > 0);
        check("RTL spike count matches Python reference expectation", spike_count_delta == EXPECTED_SPIKES);

        $display("=========================================================");
        $display("  input events                    : %0d", event_count);
        $display("  Python expected group0 spikes   : %0d", EXPECTED_SPIKES);
        $display("  RTL MNIST-style spike count     : %0d", spike_count_delta);
        $display("  MNIST-style Results             : %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("=========================================================");

        if (fail_count != 0) begin
            $display("*** MNIST HLS RTL SIM TEST FAILED ***");
            $finish(1);
        end

        $display("*** MNIST HLS RTL SIM TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #150000000;
        $display("[ERROR] MNIST-style HLS RTL simulation TB timed out");
        $finish(2);
    end

endmodule
