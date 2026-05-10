//=============================================================================
// Testbench: generated snn_top_hls learning engine only
//
// This is a simulation-only AXI-lite/AXI-stream driver for the generated HLS
// Verilog. It does not require design_1_wrapper or a board.
//=============================================================================

`timescale 1ns / 1ps

module tb_hls_learning_engine;

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

    reg ap_clk;
    reg ap_rst_n;

    initial ap_clk = 1'b0;
    always #5 ap_clk = ~ap_clk;

    // AXI-stream spike input
    reg  [31:0] s_axis_spikes_TDATA;
    reg         s_axis_spikes_TVALID;
    wire        s_axis_spikes_TREADY;
    reg  [3:0]  s_axis_spikes_TKEEP;
    reg  [3:0]  s_axis_spikes_TSTRB;
    reg  [0:0]  s_axis_spikes_TUSER;
    reg  [0:0]  s_axis_spikes_TLAST;
    reg  [0:0]  s_axis_spikes_TID;
    reg  [0:0]  s_axis_spikes_TDEST;

    // AXI-stream data input, unused here
    reg  [31:0] s_axis_data_TDATA;
    reg         s_axis_data_TVALID;
    wire        s_axis_data_TREADY;
    reg  [3:0]  s_axis_data_TKEEP;
    reg  [3:0]  s_axis_data_TSTRB;
    reg  [0:0]  s_axis_data_TUSER;
    reg  [0:0]  s_axis_data_TLAST;
    reg  [0:0]  s_axis_data_TID;
    reg  [0:0]  s_axis_data_TDEST;

    // AXI-stream weight input, unused by the main HLS-only test
    reg  [31:0] s_axis_weights_TDATA;
    reg         s_axis_weights_TVALID;
    wire        s_axis_weights_TREADY;
    reg  [3:0]  s_axis_weights_TKEEP;
    reg  [3:0]  s_axis_weights_TSTRB;
    reg  [0:0]  s_axis_weights_TUSER;
    reg  [0:0]  s_axis_weights_TLAST;
    reg  [0:0]  s_axis_weights_TID;
    reg  [0:0]  s_axis_weights_TDEST;

    // AXI-stream outputs
    wire [31:0] m_axis_spikes_TDATA;
    wire        m_axis_spikes_TVALID;
    reg         m_axis_spikes_TREADY;
    wire [3:0]  m_axis_spikes_TKEEP;
    wire [3:0]  m_axis_spikes_TSTRB;
    wire [0:0]  m_axis_spikes_TUSER;
    wire [0:0]  m_axis_spikes_TLAST;
    wire [0:0]  m_axis_spikes_TID;
    wire [0:0]  m_axis_spikes_TDEST;

    wire [31:0] m_axis_weights_TDATA;
    wire        m_axis_weights_TVALID;
    reg         m_axis_weights_TREADY;
    wire [3:0]  m_axis_weights_TKEEP;
    wire [3:0]  m_axis_weights_TSTRB;
    wire [0:0]  m_axis_weights_TUSER;
    wire [0:0]  m_axis_weights_TLAST;
    wire [0:0]  m_axis_weights_TID;
    wire [0:0]  m_axis_weights_TDEST;

    // Direct HLS/RTL spike wires
    wire [0:0]  spike_in_valid;
    wire [7:0]  spike_in_neuron_id;
    wire [7:0]  spike_in_weight;
    reg  [0:0]  spike_in_ready;
    reg  [0:0]  spike_out_valid;
    reg  [7:0]  spike_out_neuron_id;
    reg  [7:0]  spike_out_weight;
    wire [0:0]  spike_out_ready;

    wire [0:0]  snn_enable;
    wire [0:0]  snn_reset;
    wire [15:0] threshold_out;
    wire [15:0] leak_rate_out;
    reg  [0:0]  snn_ready;
    reg  [0:0]  snn_busy;

    // AXI-lite control
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

    integer fail_count;
    integer pass_count;
    integer checkpoint_seen;
    integer spike_to_rtl_seen;
    reg [7:0] initial_weight;
    reg [7:0] learned_weight;
    reg [7:0] checkpoint_weight;

    snn_top_hls dut (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .s_axis_spikes_TDATA(s_axis_spikes_TDATA),
        .s_axis_spikes_TVALID(s_axis_spikes_TVALID),
        .s_axis_spikes_TREADY(s_axis_spikes_TREADY),
        .s_axis_spikes_TKEEP(s_axis_spikes_TKEEP),
        .s_axis_spikes_TSTRB(s_axis_spikes_TSTRB),
        .s_axis_spikes_TUSER(s_axis_spikes_TUSER),
        .s_axis_spikes_TLAST(s_axis_spikes_TLAST),
        .s_axis_spikes_TID(s_axis_spikes_TID),
        .s_axis_spikes_TDEST(s_axis_spikes_TDEST),
        .s_axis_data_TDATA(s_axis_data_TDATA),
        .s_axis_data_TVALID(s_axis_data_TVALID),
        .s_axis_data_TREADY(s_axis_data_TREADY),
        .s_axis_data_TKEEP(s_axis_data_TKEEP),
        .s_axis_data_TSTRB(s_axis_data_TSTRB),
        .s_axis_data_TUSER(s_axis_data_TUSER),
        .s_axis_data_TLAST(s_axis_data_TLAST),
        .s_axis_data_TID(s_axis_data_TID),
        .s_axis_data_TDEST(s_axis_data_TDEST),
        .s_axis_weights_TDATA(s_axis_weights_TDATA),
        .s_axis_weights_TVALID(s_axis_weights_TVALID),
        .s_axis_weights_TREADY(s_axis_weights_TREADY),
        .s_axis_weights_TKEEP(s_axis_weights_TKEEP),
        .s_axis_weights_TSTRB(s_axis_weights_TSTRB),
        .s_axis_weights_TUSER(s_axis_weights_TUSER),
        .s_axis_weights_TLAST(s_axis_weights_TLAST),
        .s_axis_weights_TID(s_axis_weights_TID),
        .s_axis_weights_TDEST(s_axis_weights_TDEST),
        .m_axis_spikes_TDATA(m_axis_spikes_TDATA),
        .m_axis_spikes_TVALID(m_axis_spikes_TVALID),
        .m_axis_spikes_TREADY(m_axis_spikes_TREADY),
        .m_axis_spikes_TKEEP(m_axis_spikes_TKEEP),
        .m_axis_spikes_TSTRB(m_axis_spikes_TSTRB),
        .m_axis_spikes_TUSER(m_axis_spikes_TUSER),
        .m_axis_spikes_TLAST(m_axis_spikes_TLAST),
        .m_axis_spikes_TID(m_axis_spikes_TID),
        .m_axis_spikes_TDEST(m_axis_spikes_TDEST),
        .m_axis_weights_TDATA(m_axis_weights_TDATA),
        .m_axis_weights_TVALID(m_axis_weights_TVALID),
        .m_axis_weights_TREADY(m_axis_weights_TREADY),
        .m_axis_weights_TKEEP(m_axis_weights_TKEEP),
        .m_axis_weights_TSTRB(m_axis_weights_TSTRB),
        .m_axis_weights_TUSER(m_axis_weights_TUSER),
        .m_axis_weights_TLAST(m_axis_weights_TLAST),
        .m_axis_weights_TID(m_axis_weights_TID),
        .m_axis_weights_TDEST(m_axis_weights_TDEST),
        .spike_in_valid(spike_in_valid),
        .spike_in_neuron_id(spike_in_neuron_id),
        .spike_in_weight(spike_in_weight),
        .spike_in_ready(spike_in_ready),
        .spike_out_valid(spike_out_valid),
        .spike_out_neuron_id(spike_out_neuron_id),
        .spike_out_weight(spike_out_weight),
        .spike_out_ready(spike_out_ready),
        .snn_enable(snn_enable),
        .snn_reset(snn_reset),
        .threshold_out(threshold_out),
        .leak_rate_out(leak_rate_out),
        .snn_ready(snn_ready),
        .snn_busy(snn_busy),
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
        .interrupt(interrupt)
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
        @(posedge ap_clk);
        s_axi_ctrl_AWADDR  <= addr;
        s_axi_ctrl_WDATA   <= data;
        s_axi_ctrl_WSTRB   <= 4'hF;
        s_axi_ctrl_AWVALID <= 1'b1;
        s_axi_ctrl_WVALID  <= 1'b1;
        s_axi_ctrl_BREADY  <= 1'b1;

        timeout = 0;
        while (!s_axi_ctrl_AWREADY && timeout < 1000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        @(posedge ap_clk);
        s_axi_ctrl_AWVALID <= 1'b0;

        timeout = 0;
        while (!s_axi_ctrl_WREADY && timeout < 1000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        @(posedge ap_clk);
        s_axi_ctrl_WVALID <= 1'b0;

        timeout = 0;
        while (!s_axi_ctrl_BVALID && timeout < 1000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        @(posedge ap_clk);
        s_axi_ctrl_BREADY <= 1'b0;
    end
    endtask

    task automatic axi_read;
        input [7:0] addr;
        output [31:0] data;
        integer timeout;
    begin
        @(posedge ap_clk);
        s_axi_ctrl_ARADDR  <= addr;
        s_axi_ctrl_ARVALID <= 1'b1;
        s_axi_ctrl_RREADY  <= 1'b1;

        timeout = 0;
        while (!s_axi_ctrl_ARREADY && timeout < 1000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        @(posedge ap_clk);
        s_axi_ctrl_ARVALID <= 1'b0;

        timeout = 0;
        while (!s_axi_ctrl_RVALID && timeout < 1000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        data = s_axi_ctrl_RDATA;
        @(posedge ap_clk);
        s_axi_ctrl_RREADY <= 1'b0;
    end
    endtask

    task automatic start_kernel;
    begin
        axi_write(ADDR_AP_CTRL, 32'h0000_0001);
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

    task automatic send_pre_spike_and_wait_accept;
        input [7:0] neuron_id;
        input [7:0] weight;
        integer timeout;
    begin
        s_axis_spikes_TDATA  <= spike_word(neuron_id, weight);
        s_axis_spikes_TVALID <= 1'b1;
        s_axis_spikes_TLAST  <= 1'b1;
        timeout = 0;
        while (!(s_axis_spikes_TVALID && s_axis_spikes_TREADY) && timeout < 300000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        if (timeout >= 300000) begin
            $display("[ERROR] pre spike was not accepted");
            fail_count = fail_count + 1;
        end else begin
            $display("pre spike accepted: neuron=%0d weight=%0d time=%0t", neuron_id, weight, $time);
        end
        @(posedge ap_clk);
        s_axis_spikes_TVALID <= 1'b0;
    end
    endtask

    task automatic run_pre_kernel;
        integer done_seen;
    begin
        axi_write(ADDR_MODE_REG, MODE_TRAIN_STDP);
        axi_write(ADDR_TIME_STEPS, 32'd1);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE | CTRL_LEARNING);
        s_axis_spikes_TDATA  <= spike_word(8'd0, 8'd12);
        s_axis_spikes_TVALID <= 1'b1;
        s_axis_spikes_TLAST  <= 1'b1;
        start_kernel;
        send_pre_spike_and_wait_accept(8'd0, 8'd12);
        wait_ap_done(200000, done_seen);
        check("HLS pre-spike kernel completed", done_seen);
    end
    endtask

    task automatic run_post_kernel;
        integer done_seen;
    begin
        axi_write(ADDR_MODE_REG, MODE_TRAIN_STDP);
        axi_write(ADDR_TIME_STEPS, 32'd1);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE | CTRL_LEARNING);
        spike_out_neuron_id <= 8'd1;
        spike_out_weight    <= 8'd0;
        spike_out_valid     <= 1'b1;
        $display("teacher post spike asserted: neuron=1 time=%0t", $time);
        start_kernel;
        wait_ap_done(200000, done_seen);
        spike_out_valid <= 1'b0;
        check("HLS post-spike kernel completed", done_seen);
    end
    endtask

    task automatic run_checkpoint_kernel;
        integer done_seen;
    begin
        checkpoint_seen = 0;
        checkpoint_weight = 8'd0;
        axi_write(ADDR_MODE_REG, MODE_CHECKPOINT);
        axi_write(ADDR_TIME_STEPS, 32'd4);
        axi_write(ADDR_CTRL_REG, CTRL_ENABLE | CTRL_WEIGHT_READ);
        start_kernel;
        wait_ap_done(200000, done_seen);
        check("HLS checkpoint kernel completed", done_seen);
    end
    endtask

    always @(posedge ap_clk) begin
        if (spike_in_valid && spike_in_ready) begin
            spike_to_rtl_seen = 1;
            $display("HLS->RTL spike: neuron=%0d weight=%0d time=%0t",
                     spike_in_neuron_id, spike_in_weight, $time);
        end

        if (m_axis_weights_TVALID && m_axis_weights_TREADY) begin
            $display("HLS checkpoint word: data=0x%08h column=%0d weight=%0d last=%0d time=%0t",
                     m_axis_weights_TDATA,
                     m_axis_weights_TDATA[15:8],
                     m_axis_weights_TDATA[23:16],
                     m_axis_weights_TLAST,
                     $time);
            if (m_axis_weights_TDATA[15:8] == 8'd1) begin
                checkpoint_seen = 1;
                checkpoint_weight = m_axis_weights_TDATA[23:16];
            end
        end
    end

    initial begin
        $display("=========================================================");
        $display("  HLS generated Verilog TB: pre/post STDP smoke test");
        $display("=========================================================");

        fail_count = 0;
        pass_count = 0;
        checkpoint_seen = 0;
        spike_to_rtl_seen = 0;
        initial_weight = 0;
        learned_weight = 0;
        checkpoint_weight = 0;

        ap_rst_n = 1'b0;
        s_axis_spikes_TDATA = 0;
        s_axis_spikes_TVALID = 0;
        s_axis_spikes_TKEEP = 4'hF;
        s_axis_spikes_TSTRB = 4'hF;
        s_axis_spikes_TUSER = 0;
        s_axis_spikes_TLAST = 0;
        s_axis_spikes_TID = 0;
        s_axis_spikes_TDEST = 0;
        s_axis_data_TDATA = 0;
        s_axis_data_TVALID = 0;
        s_axis_data_TKEEP = 4'hF;
        s_axis_data_TSTRB = 4'hF;
        s_axis_data_TUSER = 0;
        s_axis_data_TLAST = 0;
        s_axis_data_TID = 0;
        s_axis_data_TDEST = 0;
        s_axis_weights_TDATA = 0;
        s_axis_weights_TVALID = 0;
        s_axis_weights_TKEEP = 4'hF;
        s_axis_weights_TSTRB = 4'hF;
        s_axis_weights_TUSER = 0;
        s_axis_weights_TLAST = 0;
        s_axis_weights_TID = 0;
        s_axis_weights_TDEST = 0;
        m_axis_spikes_TREADY = 1'b1;
        m_axis_weights_TREADY = 1'b1;
        spike_in_ready = 1'b1;
        spike_out_valid = 1'b0;
        spike_out_neuron_id = 8'd0;
        spike_out_weight = 8'd0;
        snn_ready = 1'b1;
        snn_busy = 1'b0;

        s_axi_ctrl_AWVALID = 1'b0;
        s_axi_ctrl_AWADDR = 8'd0;
        s_axi_ctrl_WVALID = 1'b0;
        s_axi_ctrl_WDATA = 32'd0;
        s_axi_ctrl_WSTRB = 4'hF;
        s_axi_ctrl_ARVALID = 1'b0;
        s_axi_ctrl_ARADDR = 8'd0;
        s_axi_ctrl_RREADY = 1'b0;
        s_axi_ctrl_BREADY = 1'b0;

        repeat (20) @(posedge ap_clk);
        ap_rst_n <= 1'b1;
        repeat (20) @(posedge ap_clk);

        axi_write(ADDR_CONFIG_REG, {16'd0, 16'd10});

        run_pre_kernel;
        repeat (20) @(posedge ap_clk);
        initial_weight = dut.p_ZL13weight_memory_1_U.ram[0];

        run_post_kernel;
        repeat (200) @(posedge ap_clk);
        learned_weight = dut.p_ZL13weight_memory_1_U.ram[0];

        run_checkpoint_kernel;
        repeat (20) @(posedge ap_clk);

        $display("initial weight         : %0d", initial_weight);
        $display("HLS learned raw weight : %0d", learned_weight);
        $display("checkpoint col1 weight : %0d", checkpoint_weight);

        check("HLS emitted pre spike on direct RTL port", spike_to_rtl_seen);
        check("HLS weight RAM changed after pre/post stimulus", learned_weight != initial_weight);
        check("checkpoint stream exposed learned weight", checkpoint_seen && checkpoint_weight == learned_weight);

        $display("=========================================================");
        $display("  HLS learning TB Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("=========================================================");

        if (fail_count != 0) begin
            $display("*** HLS LEARNING TEST FAILED ***");
            $finish(1);
        end

        $display("*** HLS LEARNING TEST PASSED ***");
        $finish(0);
    end

    initial begin
        #100000000;
        $display("[ERROR] HLS learning TB timed out");
        $finish(2);
    end

endmodule

