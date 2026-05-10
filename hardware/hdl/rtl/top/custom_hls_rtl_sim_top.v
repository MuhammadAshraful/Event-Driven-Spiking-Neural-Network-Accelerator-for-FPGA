//=============================================================================
// Simulation-only top: generated HLS snn_top_hls + RTL router/CT/core groups
//
// This file intentionally avoids design_1_wrapper. It connects the generated
// HLS Verilog to the RTL spike-processing path and adapts HLS checkpointed
// weights into the router's learned-weight port for a tiny deterministic sim.
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_hls_rtl_sim_top #(
    parameter NUM_GROUPS         = 2,
    parameter NEURONS_PER_GROUP  = 16,
    parameter WEIGHT_WIDTH       = 8,
    parameter MAX_FANOUT_INTER   = 16,
    parameter DATA_WIDTH         = 16,
    parameter THRESHOLD_WIDTH    = 16,
    parameter LEAK_WIDTH         = 8,
    parameter REFRAC_WIDTH       = 8,
    parameter SPIKE_BUFFER_DEPTH = 16,
    parameter GROUP_ID_WIDTH     = $clog2(NUM_GROUPS),
    parameter LOCAL_ID_WIDTH     = $clog2(NEURONS_PER_GROUP),
    parameter GLOBAL_ID_WIDTH    = GROUP_ID_WIDTH + LOCAL_ID_WIDTH,
    parameter FANOUT_IDX_WIDTH   = $clog2(MAX_FANOUT_INTER)
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         rtl_enable,

    // HLS AXI-stream spike input from the testbench.
    input  wire [31:0]  s_axis_spikes_TDATA,
    input  wire         s_axis_spikes_TVALID,
    output wire         s_axis_spikes_TREADY,
    input  wire [0:0]   s_axis_spikes_TLAST,

    // Teacher post spike into the HLS STDP post-spike path.
    input  wire         teacher_post_valid,
    input  wire [7:0]   teacher_post_neuron_id,
    input  wire [7:0]   teacher_post_weight,

    // Host/testbench direct core weight programming.
    input  wire [NUM_GROUPS-1:0]      host_weight_we,
    input  wire [LOCAL_ID_WIDTH-1:0]  host_weight_src,
    input  wire [LOCAL_ID_WIDTH-1:0]  host_weight_dst,
    input  wire [WEIGHT_WIDTH-1:0]    host_weight_data,
    input  wire                       host_weight_exc,

    // HLS AXI-lite control.
    input  wire         s_axi_ctrl_AWVALID,
    output wire         s_axi_ctrl_AWREADY,
    input  wire [7:0]   s_axi_ctrl_AWADDR,
    input  wire         s_axi_ctrl_WVALID,
    output wire         s_axi_ctrl_WREADY,
    input  wire [31:0]  s_axi_ctrl_WDATA,
    input  wire [3:0]   s_axi_ctrl_WSTRB,
    input  wire         s_axi_ctrl_ARVALID,
    output wire         s_axi_ctrl_ARREADY,
    input  wire [7:0]   s_axi_ctrl_ARADDR,
    output wire         s_axi_ctrl_RVALID,
    input  wire         s_axi_ctrl_RREADY,
    output wire [31:0]  s_axi_ctrl_RDATA,
    output wire [1:0]   s_axi_ctrl_RRESP,
    output wire         s_axi_ctrl_BVALID,
    input  wire         s_axi_ctrl_BREADY,
    output wire [1:0]   s_axi_ctrl_BRESP,
    output wire         interrupt,

    // Observability for testbenches.
    output wire [15:0]  g0_spike_count,
    output wire [15:0]  g1_spike_count,
    output wire         router_busy,
    output wire [NUM_GROUPS-1:0] group_busy,
    output reg  [7:0]   applied_weight_0_1,
    output reg          learned_update_valid,
    output reg  [7:0]   learned_update_weight,
    output wire [31:0]  hls_weight_stream_data,
    output wire         hls_weight_stream_valid
);

    //-------------------------------------------------------------------------
    // HLS stream/control tie-offs and direct spike wires
    //-------------------------------------------------------------------------
    wire [31:0] m_axis_spikes_TDATA;
    wire        m_axis_spikes_TVALID;
    wire [3:0]  m_axis_spikes_TKEEP;
    wire [3:0]  m_axis_spikes_TSTRB;
    wire [0:0]  m_axis_spikes_TUSER;
    wire [0:0]  m_axis_spikes_TLAST;
    wire [0:0]  m_axis_spikes_TID;
    wire [0:0]  m_axis_spikes_TDEST;

    wire [31:0] m_axis_weights_TDATA;
    wire        m_axis_weights_TVALID;
    wire [3:0]  m_axis_weights_TKEEP;
    wire [3:0]  m_axis_weights_TSTRB;
    wire [0:0]  m_axis_weights_TUSER;
    wire [0:0]  m_axis_weights_TLAST;
    wire [0:0]  m_axis_weights_TID;
    wire [0:0]  m_axis_weights_TDEST;

    wire [0:0]  hls_spike_in_valid;
    wire [7:0]  hls_spike_in_neuron_id;
    wire [7:0]  hls_spike_in_weight;
    wire [0:0]  hls_spike_in_ready;
    wire [0:0]  hls_spike_out_ready;

    wire [0:0]  hls_snn_enable;
    wire [0:0]  hls_snn_reset;
    wire [15:0] hls_threshold_out;
    wire [15:0] hls_leak_rate_out;
    wire        learn_spike_valid;
    wire [GLOBAL_ID_WIDTH-1:0] learn_spike_src_id;

    assign hls_weight_stream_data  = m_axis_weights_TDATA;
    assign hls_weight_stream_valid = m_axis_weights_TVALID;

    // The generated HLS wrapper exposes m_axis_weights as the only checked-in
    // way to observe learned weights. Keep it ready in this sim top.
    wire m_axis_weights_TREADY = 1'b1;
    wire m_axis_spikes_TREADY  = 1'b1;

    wire [0:0] hls_spike_out_valid =
        teacher_post_valid ? 1'b1 : learn_spike_valid;
    wire [7:0] hls_spike_out_neuron_id =
        teacher_post_valid ? teacher_post_neuron_id :
        {{(8-GLOBAL_ID_WIDTH){1'b0}}, learn_spike_src_id};
    wire [7:0] hls_spike_out_weight =
        teacher_post_valid ? teacher_post_weight : 8'd0;

    wire rtl_rst_n = rst_n & ~hls_snn_reset[0];
    wire core_enable = rtl_enable | hls_snn_enable[0];
    wire [THRESHOLD_WIDTH-1:0] global_threshold =
        (hls_threshold_out != 16'd0) ? hls_threshold_out : 16'd10;
    wire [LEAK_WIDTH-1:0] global_leak_rate = hls_leak_rate_out[LEAK_WIDTH-1:0];

    snn_top_hls u_hls (
        .ap_clk(clk),
        .ap_rst_n(rst_n),
        .s_axis_spikes_TDATA(s_axis_spikes_TDATA),
        .s_axis_spikes_TVALID(s_axis_spikes_TVALID),
        .s_axis_spikes_TREADY(s_axis_spikes_TREADY),
        .s_axis_spikes_TKEEP(4'hF),
        .s_axis_spikes_TSTRB(4'hF),
        .s_axis_spikes_TUSER(1'b0),
        .s_axis_spikes_TLAST(s_axis_spikes_TLAST),
        .s_axis_spikes_TID(1'b0),
        .s_axis_spikes_TDEST(1'b0),
        .s_axis_data_TDATA(32'd0),
        .s_axis_data_TVALID(1'b0),
        .s_axis_data_TREADY(),
        .s_axis_data_TKEEP(4'hF),
        .s_axis_data_TSTRB(4'hF),
        .s_axis_data_TUSER(1'b0),
        .s_axis_data_TLAST(1'b0),
        .s_axis_data_TID(1'b0),
        .s_axis_data_TDEST(1'b0),
        .s_axis_weights_TDATA(32'd0),
        .s_axis_weights_TVALID(1'b0),
        .s_axis_weights_TREADY(),
        .s_axis_weights_TKEEP(4'hF),
        .s_axis_weights_TSTRB(4'hF),
        .s_axis_weights_TUSER(1'b0),
        .s_axis_weights_TLAST(1'b0),
        .s_axis_weights_TID(1'b0),
        .s_axis_weights_TDEST(1'b0),
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
        .spike_in_valid(hls_spike_in_valid),
        .spike_in_neuron_id(hls_spike_in_neuron_id),
        .spike_in_weight(hls_spike_in_weight),
        .spike_in_ready(hls_spike_in_ready),
        .spike_out_valid(hls_spike_out_valid),
        .spike_out_neuron_id(hls_spike_out_neuron_id),
        .spike_out_weight(hls_spike_out_weight),
        .spike_out_ready(hls_spike_out_ready),
        .snn_enable(hls_snn_enable),
        .snn_reset(hls_snn_reset),
        .threshold_out(hls_threshold_out),
        .leak_rate_out(hls_leak_rate_out),
        .snn_ready(~router_busy),
        .snn_busy(router_busy | (|group_busy)),
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

    //-------------------------------------------------------------------------
    // RTL core groups, router, and connectivity table
    //-------------------------------------------------------------------------
    wire [NUM_GROUPS-1:0]                grp_spike_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0] grp_spike_neuron_id;
    wire [NUM_GROUPS-1:0]                grp_spike_ready;
    wire [NUM_GROUPS-1:0]                grp_in_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0] grp_in_dest_id;
    wire [NUM_GROUPS*WEIGHT_WIDTH-1:0]   grp_in_weight;
    wire [NUM_GROUPS-1:0]                grp_in_exc;
    wire [NUM_GROUPS-1:0]                grp_in_ready;

    wire [NUM_GROUPS-1:0]                grp_weight_we;
    wire [LOCAL_ID_WIDTH-1:0]            grp_weight_src;
    wire [LOCAL_ID_WIDTH-1:0]            grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0]              grp_weight_data;
    wire                                 grp_weight_exc;
    wire [NUM_GROUPS*16-1:0]             grp_spike_count;

    wire                                 ct_lookup_en;
    wire [GROUP_ID_WIDTH-1:0]            ct_lookup_src_group;
    wire [LOCAL_ID_WIDTH-1:0]            ct_lookup_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0]          ct_lookup_fanout_idx;
    wire                                 ct_result_valid;
    wire [GROUP_ID_WIDTH-1:0]            ct_result_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]            ct_result_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]              ct_result_weight;
    wire                                 ct_result_exc_inh;
    wire                                 ct_result_entry_valid;

    wire                                 ct_cfg_we;
    wire [GROUP_ID_WIDTH-1:0]            ct_cfg_src_group;
    wire [LOCAL_ID_WIDTH-1:0]            ct_cfg_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0]          ct_cfg_fanout_idx;
    wire                                 ct_cfg_valid;
    wire [GROUP_ID_WIDTH-1:0]            ct_cfg_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]            ct_cfg_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]              ct_cfg_weight;
    wire                                 ct_cfg_exc_inh;

    wire                                 learn_weight_ready;
    wire                                 router_ext_ready;

    // The checked-in generated HLS wrapper samples this ap_none ready input
    // early in the kernel schedule. Keep it asserted in the sim top so the
    // AXI-stream pre spike is consumed. The direct HLS ap_none valid can remain
    // level-high across many cycles, so this sim top forwards exactly one RTL
    // router event when the generated HLS AXI-stream input handshakes.
    assign hls_spike_in_ready = 1'b1;

    reg                       hls_ext_pending_valid;
    reg [GLOBAL_ID_WIDTH-1:0] hls_ext_pending_id;
    reg [WEIGHT_WIDTH-1:0]    hls_ext_pending_weight;
    reg                       hls_ext_pending_exc;
    reg                       hls_stream_seen;

    wire hls_stream_spike_hs = s_axis_spikes_TVALID & s_axis_spikes_TREADY;
    wire hls_stream_weight_negative = s_axis_spikes_TDATA[17];
    wire [WEIGHT_WIDTH-1:0] hls_stream_weight_raw = s_axis_spikes_TDATA[17:10];
    wire [WEIGHT_WIDTH-1:0] hls_stream_weight_mag =
        hls_stream_weight_negative ?
        (~hls_stream_weight_raw + {{(WEIGHT_WIDTH-1){1'b0}}, 1'b1}) :
        hls_stream_weight_raw;

    always @(posedge clk) begin
        if (!rtl_rst_n) begin
            hls_ext_pending_valid <= 1'b0;
            hls_ext_pending_id    <= {GLOBAL_ID_WIDTH{1'b0}};
            hls_ext_pending_weight <= {WEIGHT_WIDTH{1'b0}};
            hls_ext_pending_exc   <= 1'b1;
            hls_stream_seen       <= 1'b0;
        end else begin
            if (!s_axis_spikes_TVALID)
                hls_stream_seen <= 1'b0;

            if (hls_ext_pending_valid && router_ext_ready)
                hls_ext_pending_valid <= 1'b0;

            if (hls_stream_spike_hs && !hls_stream_seen && !hls_ext_pending_valid) begin
                hls_stream_seen       <= 1'b1;
                hls_ext_pending_valid <= 1'b1;
                hls_ext_pending_id    <= s_axis_spikes_TDATA[GLOBAL_ID_WIDTH-1:0];
                hls_ext_pending_weight <= hls_stream_weight_mag;
                hls_ext_pending_exc   <= ~hls_stream_weight_negative;
            end else if (hls_stream_spike_hs && !hls_stream_seen) begin
                hls_stream_seen <= 1'b1;
            end
        end
    end

    // Checkpoint-to-learned-weight adapter for the tiny experiment:
    // checkpoint column 1 corresponds to generated-HLS RAM bank1/address0,
    // i.e. legacy dense synapse pre0 -> post1.
    reg        pending_learn_weight_valid;
    reg [7:0]  pending_learn_weight_data;
    wire       hls_weight_hs = m_axis_weights_TVALID & m_axis_weights_TREADY;
    wire       hls_checkpoint_col1 = hls_weight_hs & (m_axis_weights_TDATA[15:8] == 8'd1);

    always @(posedge clk) begin
        if (!rtl_rst_n) begin
            pending_learn_weight_valid <= 1'b0;
            pending_learn_weight_data  <= 8'd0;
            learned_update_valid       <= 1'b0;
            learned_update_weight      <= 8'd0;
            applied_weight_0_1         <= 8'd0;
        end else begin
            learned_update_valid <= 1'b0;

            if (host_weight_we[0] && host_weight_src == {LOCAL_ID_WIDTH{1'b0}} &&
                host_weight_dst == {{(LOCAL_ID_WIDTH-1){1'b0}}, 1'b1}) begin
                applied_weight_0_1 <= host_weight_data;
            end

            if (hls_checkpoint_col1) begin
                pending_learn_weight_valid <= 1'b1;
                pending_learn_weight_data  <= m_axis_weights_TDATA[23:16];
                learned_update_valid       <= 1'b1;
                learned_update_weight      <= m_axis_weights_TDATA[23:16];
            end else if (pending_learn_weight_valid && learn_weight_ready) begin
                pending_learn_weight_valid <= 1'b0;
                applied_weight_0_1 <= pending_learn_weight_data;
            end
        end
    end

    wire [NUM_GROUPS-1:0] combined_weight_we;
    wire [LOCAL_ID_WIDTH-1:0] combined_weight_src [0:NUM_GROUPS-1];
    wire [LOCAL_ID_WIDTH-1:0] combined_weight_dst [0:NUM_GROUPS-1];
    wire [WEIGHT_WIDTH-1:0]   combined_weight_data[0:NUM_GROUPS-1];
    wire                      combined_weight_exc [0:NUM_GROUPS-1];

    genvar g;
    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_weight_mux
            assign combined_weight_we[g]   = host_weight_we[g] | grp_weight_we[g];
            assign combined_weight_src[g]  = host_weight_we[g] ? host_weight_src  : grp_weight_src;
            assign combined_weight_dst[g]  = host_weight_we[g] ? host_weight_dst  : grp_weight_dst;
            assign combined_weight_data[g] = host_weight_we[g] ? host_weight_data : grp_weight_data;
            assign combined_weight_exc[g]  = host_weight_we[g] ? host_weight_exc  : grp_weight_exc;

            core_group #(
                .GROUP_ID(g),
                .NEURONS_PER_GROUP(NEURONS_PER_GROUP),
                .DATA_WIDTH(DATA_WIDTH),
                .WEIGHT_WIDTH(WEIGHT_WIDTH),
                .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
                .LEAK_WIDTH(LEAK_WIDTH),
                .REFRAC_WIDTH(REFRAC_WIDTH),
                .SPIKE_BUFFER_DEPTH(SPIKE_BUFFER_DEPTH)
            ) u_core_group (
                .clk(clk),
                .rst_n(rtl_rst_n),
                .enable(core_enable),
                .ext_spike_valid(grp_in_valid[g]),
                .ext_spike_dest_id(grp_in_dest_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .ext_spike_weight(grp_in_weight[g*WEIGHT_WIDTH +: WEIGHT_WIDTH]),
                .ext_spike_exc_inh(grp_in_exc[g]),
                .ext_spike_ready(grp_in_ready[g]),
                .out_spike_valid(grp_spike_valid[g]),
                .out_spike_neuron_id(grp_spike_neuron_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .out_spike_ready(grp_spike_ready[g]),
                .global_threshold(global_threshold),
                .global_leak_rate(global_leak_rate),
                .global_refrac_period(8'd3),
                .weight_we(combined_weight_we[g]),
                .weight_src_id(combined_weight_src[g]),
                .weight_dst_id(combined_weight_dst[g]),
                .weight_data(combined_weight_data[g]),
                .weight_exc(combined_weight_exc[g]),
                .spike_count(grp_spike_count[g*16 +: 16]),
                .group_busy(group_busy[g])
            );
        end
    endgenerate

    synaptic_connectivity_table #(
        .NUM_GROUPS(NUM_GROUPS),
        .NEURONS_PER_GROUP(NEURONS_PER_GROUP),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_FANOUT_INTER(MAX_FANOUT_INTER)
    ) u_ct (
        .clk(clk),
        .rst_n(rtl_rst_n),
        .cfg_we(ct_cfg_we),
        .cfg_src_group(ct_cfg_src_group),
        .cfg_src_neuron(ct_cfg_src_neuron),
        .cfg_fanout_idx(ct_cfg_fanout_idx),
        .cfg_valid(ct_cfg_valid),
        .cfg_dst_group(ct_cfg_dst_group),
        .cfg_dst_neuron(ct_cfg_dst_neuron),
        .cfg_weight(ct_cfg_weight),
        .cfg_exc_inh(ct_cfg_exc_inh),
        .lookup_en(ct_lookup_en),
        .lookup_src_group(ct_lookup_src_group),
        .lookup_src_neuron(ct_lookup_src_neuron),
        .lookup_fanout_idx(ct_lookup_fanout_idx),
        .result_valid(ct_result_valid),
        .result_dst_group(ct_result_dst_group),
        .result_dst_neuron(ct_result_dst_neuron),
        .result_weight(ct_result_weight),
        .result_exc_inh(ct_result_exc_inh),
        .result_entry_valid(ct_result_entry_valid)
    );

    event_router_ng #(
        .NUM_GROUPS(NUM_GROUPS),
        .NEURONS_PER_GROUP(NEURONS_PER_GROUP),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_FANOUT_INTER(MAX_FANOUT_INTER)
    ) u_router (
        .clk(clk),
        .rst_n(rtl_rst_n),
        .enable(core_enable),
        .grp_spike_valid(grp_spike_valid),
        .grp_spike_neuron_id(grp_spike_neuron_id),
        .grp_spike_ready(grp_spike_ready),
        .grp_in_valid(grp_in_valid),
        .grp_in_dest_id(grp_in_dest_id),
        .grp_in_weight(grp_in_weight),
        .grp_in_exc(grp_in_exc),
        .grp_in_ready(grp_in_ready),
        .ext_spike_valid(hls_ext_pending_valid),
        .ext_spike_neuron_id(hls_ext_pending_id),
        .ext_spike_weight(hls_ext_pending_weight),
        .ext_spike_exc(hls_ext_pending_exc),
        .ext_spike_ready(router_ext_ready),
        .learn_spike_valid(learn_spike_valid),
        .learn_spike_src_id(learn_spike_src_id),
        .learn_spike_ready(1'b1),
        .learn_weight_valid(pending_learn_weight_valid),
        .learn_weight_group({GROUP_ID_WIDTH{1'b0}}),
        .learn_weight_src({LOCAL_ID_WIDTH{1'b0}}),
        .learn_weight_dst({{(LOCAL_ID_WIDTH-1){1'b0}}, 1'b1}),
        .learn_weight_data(pending_learn_weight_data),
        .learn_weight_exc(1'b1),
        .learn_weight_is_inter(1'b0),
        .learn_weight_dst_group({GROUP_ID_WIDTH{1'b0}}),
        .learn_weight_fanout_idx({FANOUT_IDX_WIDTH{1'b0}}),
        .learn_weight_ready(learn_weight_ready),
        .ct_lookup_en(ct_lookup_en),
        .ct_lookup_src_group(ct_lookup_src_group),
        .ct_lookup_src_neuron(ct_lookup_src_neuron),
        .ct_lookup_fanout_idx(ct_lookup_fanout_idx),
        .ct_result_valid(ct_result_valid),
        .ct_result_dst_group(ct_result_dst_group),
        .ct_result_dst_neuron(ct_result_dst_neuron),
        .ct_result_weight(ct_result_weight),
        .ct_result_exc_inh(ct_result_exc_inh),
        .ct_result_entry_valid(ct_result_entry_valid),
        .grp_weight_we(grp_weight_we),
        .grp_weight_src(grp_weight_src),
        .grp_weight_dst(grp_weight_dst),
        .grp_weight_data(grp_weight_data),
        .grp_weight_exc(grp_weight_exc),
        .ct_cfg_we(ct_cfg_we),
        .ct_cfg_src_group(ct_cfg_src_group),
        .ct_cfg_src_neuron(ct_cfg_src_neuron),
        .ct_cfg_fanout_idx(ct_cfg_fanout_idx),
        .ct_cfg_valid(ct_cfg_valid),
        .ct_cfg_dst_group(ct_cfg_dst_group),
        .ct_cfg_dst_neuron(ct_cfg_dst_neuron),
        .ct_cfg_weight(ct_cfg_weight),
        .ct_cfg_exc_inh(ct_cfg_exc_inh),
        .routed_spike_count(),
        .router_busy(router_busy)
    );

    assign g0_spike_count = grp_spike_count[0*16 +: 16];
    assign g1_spike_count = grp_spike_count[1*16 +: 16];

endmodule
