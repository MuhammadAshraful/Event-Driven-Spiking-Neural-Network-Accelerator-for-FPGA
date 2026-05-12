//=============================================================================
// RTL-only unsupervised STDP simulation top
//
// No HLS, no AXI control, no design_1_wrapper, and no teacher post spike.
// Input neurons fire from external spike events. The output neuron fires only
// when accumulated recurrent input-to-output weights cross threshold.
//
// Demonstration network inside group 0:
//   input neurons : 0, 1, 2
//   output neuron : 3
//   learned paths : 0->3, 1->3, 2->3
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_rtl_unsupervised_sim_top #(
    parameter NUM_GROUPS         = 2,
    parameter NEURONS_PER_GROUP  = 16,
    parameter WEIGHT_WIDTH       = 8,
    parameter MAX_FANOUT_INTER   = 16,
    parameter DATA_WIDTH         = 16,
    parameter THRESHOLD_WIDTH    = 16,
    parameter LEAK_WIDTH         = 8,
    parameter REFRAC_WIDTH       = 8,
    parameter SPIKE_BUFFER_DEPTH = 32,
    parameter TIME_WIDTH         = 16,
    parameter STDP_WINDOW        = 20000,
    parameter A_PLUS             = 3,
    parameter A_MINUS            = 0,
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
    input  wire [LEAK_WIDTH-1:0]        global_leak_rate,

    input  wire                         ext_spike_valid,
    input  wire [GLOBAL_ID_WIDTH-1:0]   ext_spike_neuron_id,
    input  wire [WEIGHT_WIDTH-1:0]      ext_spike_weight,
    input  wire                         ext_spike_exc,
    output wire                         ext_spike_ready,

    input  wire [NUM_GROUPS-1:0]        host_weight_we,
    input  wire [LOCAL_ID_WIDTH-1:0]    host_weight_src,
    input  wire [LOCAL_ID_WIDTH-1:0]    host_weight_dst,
    input  wire [WEIGHT_WIDTH-1:0]      host_weight_data,
    input  wire                         host_weight_exc,

    output wire [15:0]                  g0_spike_count,
    output wire                         router_busy,
    output wire [NUM_GROUPS-1:0]        group_busy,
    output wire [TIME_WIDTH-1:0]        current_time_out,
    output reg  [WEIGHT_WIDTH-1:0]      weight_0_3,
    output reg  [WEIGHT_WIDTH-1:0]      weight_1_3,
    output reg  [WEIGHT_WIDTH-1:0]      weight_2_3,
    output reg  [15:0]                  output_spike_count,
    output reg  [15:0]                  learned_update_count,
    output reg                          learned_update_pulse,
    output reg  [LOCAL_ID_WIDTH-1:0]    learned_update_src,
    output reg  [WEIGHT_WIDTH-1:0]      learned_update_weight,
    output wire                         natural_output_observed
);

    localparam [LOCAL_ID_WIDTH-1:0] INPUT0 = 4'd0;
    localparam [LOCAL_ID_WIDTH-1:0] INPUT1 = 4'd1;
    localparam [LOCAL_ID_WIDTH-1:0] INPUT2 = 4'd2;
    localparam [LOCAL_ID_WIDTH-1:0] OUTPUT = 4'd3;

    reg [TIME_WIDTH-1:0] current_time;
    assign current_time_out = current_time;

    always @(posedge clk) begin
        if (!rst_n)
            current_time <= {TIME_WIDTH{1'b0}};
        else if (enable)
            current_time <= current_time + {{(TIME_WIDTH-1){1'b0}}, 1'b1};
    end

    //-------------------------------------------------------------------------
    // Core/router/CT wiring
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
    wire learn_group0 = (learn_group == {GROUP_ID_WIDTH{1'b0}});

    wire pre0_seen = learn_spike_hs && learn_group0 && (learn_local == INPUT0);
    wire pre1_seen = learn_spike_hs && learn_group0 && (learn_local == INPUT1);
    wire pre2_seen = learn_spike_hs && learn_group0 && (learn_local == INPUT2);
    assign natural_output_observed =
        learn_spike_hs && learn_group0 && (learn_local == OUTPUT);

    always @(posedge clk) begin
        if (!rst_n)
            output_spike_count <= 16'd0;
        else if (natural_output_observed)
            output_spike_count <= output_spike_count + 16'd1;
    end

    //-------------------------------------------------------------------------
    // Three tiny STDP learners, one for each input->output synapse.
    //-------------------------------------------------------------------------
    wire [2:0] eng_update_valid;
    wire [2:0] eng_update_ready;
    wire [LOCAL_ID_WIDTH-1:0] eng_update_src [0:2];
    wire [LOCAL_ID_WIDTH-1:0] eng_update_dst [0:2];
    wire [WEIGHT_WIDTH-1:0] eng_update_weight [0:2];
    wire signed [WEIGHT_WIDTH:0] eng_delta [0:2];
    wire [1:0] eng_rule [0:2];

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
    ) u_stdp_0 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable && stdp_enable),
        .current_time(current_time),
        .pre_spike_valid(pre0_seen),
        .pre_neuron_id(INPUT0),
        .post_spike_valid(natural_output_observed),
        .post_neuron_id(OUTPUT),
        .current_weight(weight_0_3),
        .update_ready(eng_update_ready[0]),
        .update_valid(eng_update_valid[0]),
        .update_src(eng_update_src[0]),
        .update_dst(eng_update_dst[0]),
        .update_weight(eng_update_weight[0]),
        .update_exc(),
        .debug_delta(eng_delta[0]),
        .debug_rule_applied(eng_rule[0])
    );

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
    ) u_stdp_1 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable && stdp_enable),
        .current_time(current_time),
        .pre_spike_valid(pre1_seen),
        .pre_neuron_id(INPUT1),
        .post_spike_valid(natural_output_observed),
        .post_neuron_id(OUTPUT),
        .current_weight(weight_1_3),
        .update_ready(eng_update_ready[1]),
        .update_valid(eng_update_valid[1]),
        .update_src(eng_update_src[1]),
        .update_dst(eng_update_dst[1]),
        .update_weight(eng_update_weight[1]),
        .update_exc(),
        .debug_delta(eng_delta[1]),
        .debug_rule_applied(eng_rule[1])
    );

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
    ) u_stdp_2 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable && stdp_enable),
        .current_time(current_time),
        .pre_spike_valid(pre2_seen),
        .pre_neuron_id(INPUT2),
        .post_spike_valid(natural_output_observed),
        .post_neuron_id(OUTPUT),
        .current_weight(weight_2_3),
        .update_ready(eng_update_ready[2]),
        .update_valid(eng_update_valid[2]),
        .update_src(eng_update_src[2]),
        .update_dst(eng_update_dst[2]),
        .update_weight(eng_update_weight[2]),
        .update_exc(),
        .debug_delta(eng_delta[2]),
        .debug_rule_applied(eng_rule[2])
    );

    wire sel0 = eng_update_valid[0];
    wire sel1 = !sel0 && eng_update_valid[1];
    wire sel2 = !sel0 && !sel1 && eng_update_valid[2];
    wire selected_update_valid = sel0 || sel1 || sel2;
    wire [LOCAL_ID_WIDTH-1:0] selected_src =
        sel0 ? eng_update_src[0] : (sel1 ? eng_update_src[1] : eng_update_src[2]);
    wire [WEIGHT_WIDTH-1:0] selected_weight =
        sel0 ? eng_update_weight[0] : (sel1 ? eng_update_weight[1] : eng_update_weight[2]);

    assign eng_update_ready[0] = sel0 && learn_weight_ready;
    assign eng_update_ready[1] = sel1 && learn_weight_ready;
    assign eng_update_ready[2] = sel2 && learn_weight_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_0_3 <= {WEIGHT_WIDTH{1'b0}};
            weight_1_3 <= {WEIGHT_WIDTH{1'b0}};
            weight_2_3 <= {WEIGHT_WIDTH{1'b0}};
            learned_update_count <= 16'd0;
            learned_update_pulse <= 1'b0;
            learned_update_src <= {LOCAL_ID_WIDTH{1'b0}};
            learned_update_weight <= {WEIGHT_WIDTH{1'b0}};
        end else begin
            learned_update_pulse <= 1'b0;

            if (host_weight_we[0] && host_weight_dst == OUTPUT) begin
                if (host_weight_src == INPUT0)
                    weight_0_3 <= host_weight_data;
                else if (host_weight_src == INPUT1)
                    weight_1_3 <= host_weight_data;
                else if (host_weight_src == INPUT2)
                    weight_2_3 <= host_weight_data;
            end

            if (selected_update_valid && learn_weight_ready) begin
                learned_update_count <= learned_update_count + 16'd1;
                learned_update_pulse <= 1'b1;
                learned_update_src <= selected_src;
                learned_update_weight <= selected_weight;

                if (sel0)
                    weight_0_3 <= eng_update_weight[0];
                else if (sel1)
                    weight_1_3 <= eng_update_weight[1];
                else if (sel2)
                    weight_2_3 <= eng_update_weight[2];
            end
        end
    end

    //-------------------------------------------------------------------------
    // Weight mux and instantiated RTL SNN subsystem.
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
        .learn_weight_valid(selected_update_valid),
        .learn_weight_group({GROUP_ID_WIDTH{1'b0}}),
        .learn_weight_src(selected_src),
        .learn_weight_dst(OUTPUT),
        .learn_weight_data(selected_weight),
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

endmodule
