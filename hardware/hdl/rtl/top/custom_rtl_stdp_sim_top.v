//=============================================================================
// RTL-only STDP simulation top
//
// This top intentionally avoids HLS, AXI, design_1_wrapper, and board logic.
// It connects:
//   external testbench spikes -> event_router_ng -> core_group
//   core_group output spikes  -> event_router_ng learn observation
//   learn observation         -> stdp_learning_engine
//   STDP weight update        -> event_router_ng learned-weight interface
//
// The demo tracks one synapse: group0 neuron0 -> group0 neuron1.
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_rtl_stdp_sim_top #(
    parameter NUM_GROUPS         = 2,
    parameter NEURONS_PER_GROUP  = 16,
    parameter WEIGHT_WIDTH       = 8,
    parameter MAX_FANOUT_INTER   = 16,
    parameter DATA_WIDTH         = 16,
    parameter THRESHOLD_WIDTH    = 16,
    parameter LEAK_WIDTH         = 8,
    parameter REFRAC_WIDTH       = 8,
    parameter SPIKE_BUFFER_DEPTH = 16,
    parameter TIME_WIDTH         = 16,
    parameter STDP_WINDOW        = 20000,
    parameter A_PLUS             = 5,
    parameter A_MINUS            = 3,
    parameter W_MIN              = 0,
    parameter W_MAX              = 15,
    parameter GROUP_ID_WIDTH     = $clog2(NUM_GROUPS),
    parameter LOCAL_ID_WIDTH     = $clog2(NEURONS_PER_GROUP),
    parameter GLOBAL_ID_WIDTH    = GROUP_ID_WIDTH + LOCAL_ID_WIDTH,
    parameter FANOUT_IDX_WIDTH   = $clog2(MAX_FANOUT_INTER)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    input  wire                         stdp_enable,

    // Testbench-driven external spike input.
    input  wire                         ext_spike_valid,
    input  wire [GLOBAL_ID_WIDTH-1:0]   ext_spike_neuron_id,
    input  wire [WEIGHT_WIDTH-1:0]      ext_spike_weight,
    input  wire                         ext_spike_exc,
    output wire                         ext_spike_ready,

    // Testbench/direct initial local weight programming.
    input  wire [NUM_GROUPS-1:0]        host_weight_we,
    input  wire [LOCAL_ID_WIDTH-1:0]    host_weight_src,
    input  wire [LOCAL_ID_WIDTH-1:0]    host_weight_dst,
    input  wire [WEIGHT_WIDTH-1:0]      host_weight_data,
    input  wire                         host_weight_exc,

    // Observability.
    output wire [15:0]                  g0_spike_count,
    output wire [15:0]                  g1_spike_count,
    output wire                         router_busy,
    output wire [NUM_GROUPS-1:0]        group_busy,
    output wire [TIME_WIDTH-1:0]        current_time_out,
    output reg  [WEIGHT_WIDTH-1:0]      tracked_weight_0_1,
    output reg                          learned_update_pulse,
    output wire                         stdp_update_valid,
    output wire [LOCAL_ID_WIDTH-1:0]    stdp_update_src,
    output wire [LOCAL_ID_WIDTH-1:0]    stdp_update_dst,
    output wire [WEIGHT_WIDTH-1:0]      stdp_update_weight,
    output wire signed [WEIGHT_WIDTH:0] stdp_debug_delta,
    output wire [1:0]                   stdp_debug_rule_applied,
    output wire                         stdp_pre_observed,
    output wire                         stdp_post_observed
);

    localparam [LOCAL_ID_WIDTH-1:0] TRACK_SRC = {LOCAL_ID_WIDTH{1'b0}};
    localparam [LOCAL_ID_WIDTH-1:0] TRACK_DST = {{(LOCAL_ID_WIDTH-1){1'b0}}, 1'b1};

    reg [TIME_WIDTH-1:0] current_time;
    assign current_time_out = current_time;

    always @(posedge clk) begin
        if (!rst_n)
            current_time <= {TIME_WIDTH{1'b0}};
        else if (enable)
            current_time <= current_time + {{(TIME_WIDTH-1){1'b0}}, 1'b1};
    end

    //-------------------------------------------------------------------------
    // Core groups, router, and connectivity table wires
    //-------------------------------------------------------------------------
    wire [NUM_GROUPS-1:0]                grp_spike_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0] grp_spike_neuron_id;
    wire [NUM_GROUPS-1:0]                grp_spike_ready;
    wire [NUM_GROUPS-1:0]                grp_in_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0] grp_in_dest_id;
    wire [NUM_GROUPS*WEIGHT_WIDTH-1:0]   grp_in_weight;
    wire [NUM_GROUPS-1:0]                grp_in_exc;
    wire [NUM_GROUPS-1:0]                grp_in_ready;
    wire [NUM_GROUPS*16-1:0]             grp_spike_count;

    wire [NUM_GROUPS-1:0]                grp_weight_we;
    wire [LOCAL_ID_WIDTH-1:0]            grp_weight_src;
    wire [LOCAL_ID_WIDTH-1:0]            grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0]              grp_weight_data;
    wire                                 grp_weight_exc;

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

    wire                                 learn_spike_valid;
    wire [GLOBAL_ID_WIDTH-1:0]           learn_spike_src_id;
    wire                                 learn_spike_ready = 1'b1;
    wire                                 learn_weight_ready;
    wire [31:0]                          routed_spike_count_unused;

    wire [GROUP_ID_WIDTH-1:0] learn_group =
        learn_spike_src_id[GLOBAL_ID_WIDTH-1:LOCAL_ID_WIDTH];
    wire [LOCAL_ID_WIDTH-1:0] learn_local =
        learn_spike_src_id[LOCAL_ID_WIDTH-1:0];
    wire learn_spike_hs = learn_spike_valid & learn_spike_ready;

    assign stdp_pre_observed =
        learn_spike_hs && (learn_group == {GROUP_ID_WIDTH{1'b0}}) &&
        (learn_local == TRACK_SRC);
    assign stdp_post_observed =
        learn_spike_hs && (learn_group == {GROUP_ID_WIDTH{1'b0}}) &&
        (learn_local == TRACK_DST);

    //-------------------------------------------------------------------------
    // STDP engine
    //-------------------------------------------------------------------------
    stdp_learning_engine #(
        .NUM_NEURONS(NEURONS_PER_GROUP),
        .NEURON_ID_WIDTH(LOCAL_ID_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .TIME_WIDTH(TIME_WIDTH),
        .STDP_WINDOW(STDP_WINDOW),
        .A_PLUS(A_PLUS),
        .A_MINUS(A_MINUS),
        .W_MIN(W_MIN),
        .W_MAX(W_MAX)
    ) u_stdp (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable && stdp_enable),
        .current_time(current_time),
        .pre_spike_valid(stdp_pre_observed),
        .pre_neuron_id(TRACK_SRC),
        .post_spike_valid(stdp_post_observed),
        .post_neuron_id(TRACK_DST),
        .current_weight(tracked_weight_0_1),
        .update_ready(learn_weight_ready),
        .update_valid(stdp_update_valid),
        .update_src(stdp_update_src),
        .update_dst(stdp_update_dst),
        .update_weight(stdp_update_weight),
        .update_exc(),
        .debug_delta(stdp_debug_delta),
        .debug_rule_applied(stdp_debug_rule_applied)
    );

    // Track the demonstrated n0->n1 weight in a tiny readable register. The
    // actual RTL write is still performed through event_router_ng below.
    always @(posedge clk) begin
        if (!rst_n) begin
            tracked_weight_0_1  <= {WEIGHT_WIDTH{1'b0}};
            learned_update_pulse <= 1'b0;
        end else begin
            learned_update_pulse <= 1'b0;

            if (host_weight_we[0] && host_weight_src == TRACK_SRC &&
                host_weight_dst == TRACK_DST) begin
                tracked_weight_0_1 <= host_weight_data;
            end

            if (stdp_update_valid && learn_weight_ready &&
                stdp_update_src == TRACK_SRC && stdp_update_dst == TRACK_DST) begin
                tracked_weight_0_1  <= stdp_update_weight;
                learned_update_pulse <= 1'b1;
            end
        end
    end

    //-------------------------------------------------------------------------
    // Weight write mux: host programming or router-forwarded learned update.
    //-------------------------------------------------------------------------
    wire [NUM_GROUPS-1:0]     combined_weight_we;
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
                .rst_n(rst_n),
                .enable(enable),
                .ext_spike_valid(grp_in_valid[g]),
                .ext_spike_dest_id(grp_in_dest_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .ext_spike_weight(grp_in_weight[g*WEIGHT_WIDTH +: WEIGHT_WIDTH]),
                .ext_spike_exc_inh(grp_in_exc[g]),
                .ext_spike_ready(grp_in_ready[g]),
                .out_spike_valid(grp_spike_valid[g]),
                .out_spike_neuron_id(grp_spike_neuron_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .out_spike_ready(grp_spike_ready[g]),
                .global_threshold(16'd10),
                .global_leak_rate(8'd0),
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
        .rst_n(rst_n),
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
        .rst_n(rst_n),
        .enable(enable),
        .grp_spike_valid(grp_spike_valid),
        .grp_spike_neuron_id(grp_spike_neuron_id),
        .grp_spike_ready(grp_spike_ready),
        .grp_in_valid(grp_in_valid),
        .grp_in_dest_id(grp_in_dest_id),
        .grp_in_weight(grp_in_weight),
        .grp_in_exc(grp_in_exc),
        .grp_in_ready(grp_in_ready),
        .ext_spike_valid(ext_spike_valid),
        .ext_spike_neuron_id(ext_spike_neuron_id),
        .ext_spike_weight(ext_spike_weight),
        .ext_spike_exc(ext_spike_exc),
        .ext_spike_ready(ext_spike_ready),
        .learn_spike_valid(learn_spike_valid),
        .learn_spike_src_id(learn_spike_src_id),
        .learn_spike_ready(learn_spike_ready),
        .learn_weight_valid(stdp_update_valid),
        .learn_weight_group({GROUP_ID_WIDTH{1'b0}}),
        .learn_weight_src(stdp_update_src),
        .learn_weight_dst(stdp_update_dst),
        .learn_weight_data(stdp_update_weight),
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
        .routed_spike_count(routed_spike_count_unused),
        .router_busy(router_busy)
    );

    assign g0_spike_count = grp_spike_count[0*16 +: 16];
    assign g1_spike_count = grp_spike_count[1*16 +: 16];

endmodule
