//=============================================================================
// Two-core_group + CT MNIST classifier top
//
// Architecture milestone:
//   MNIST events -> event_router_ng external input -> core_group 0
//   core_group 0 natural spikes -> event_router_ng -> synaptic_connectivity_table
//   CT fanout -> event_router_ng -> core_group 1
//   core_group 1 natural spikes -> WTA/STDP
//   learned inter-group weights -> event_router_ng.learn_weight_* -> CT
//
// The classifier keeps a shadow CT weight table only because the CT has no
// readback port. Spike processing uses synaptic_connectivity_table entries.
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_rtl_mnist_twogroup_ct_classifier_top #(
    parameter INPUT_NEURONS      = 64,
    parameter OUTPUT_NEURONS     = 10,
    parameter CORE_NEURONS       = 128,
    parameter INPUT_ID_WIDTH     = 6,
    parameter OUTPUT_ID_WIDTH    = 4,
    parameter CORE_ID_WIDTH      = 7,
    parameter WEIGHT_WIDTH       = 8,
    parameter EVENT_WEIGHT_WIDTH = 4,
    parameter TIME_WIDTH         = 32,
    parameter W_MIN              = 0,
    parameter W_MAX              = 15,
    parameter A_PLUS             = 2,
    parameter OUTPUT_THRESHOLD   = 96,
    parameter INPUT_THRESHOLD    = 8,
    parameter DRAIN_CYCLES       = 1600,
    parameter WTA_INHIBIT_CYCLES = 512
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          enable,
    input  wire                          learning_enable,

    input  wire                          image_start,
    input  wire                          event_valid,
    input  wire [INPUT_ID_WIDTH-1:0]     event_neuron_id,
    input  wire [EVENT_WEIGHT_WIDTH-1:0] event_weight,
    input  wire                          image_end,
    output wire                          event_ready,

    output reg                           init_done,
    output reg                           image_done,
    output reg                           winner_valid,
    output reg  [OUTPUT_ID_WIDTH-1:0]    winner_id,
    output reg  [TIME_WIDTH-1:0]         latency_cycles,
    output reg  [15:0]                   image_output_spike_count,

    output reg  [31:0]                   total_images,
    output reg  [31:0]                   total_input_spikes,
    output reg  [31:0]                   total_group0_spikes,
    output reg  [31:0]                   total_group1_spikes,
    output reg  [31:0]                   total_ct_changed_weights,
    output reg  [WEIGHT_WIDTH-1:0]       debug_weight_min,
    output reg  [WEIGHT_WIDTH-1:0]       debug_weight_max,
    output reg  [31:0]                   debug_weight_sum,

    output wire [31:0]                   router_routed_spike_count,
    output reg  [31:0]                   router_observed_spike_count,
    output reg  [31:0]                   inter_group_routed_spikes,
    output reg  [31:0]                   ct_init_write_count,
    output reg  [31:0]                   ct_learned_update_count,
    output wire [31:0]                   direct_learning_write_count,
    output wire [31:0]                   direct_ct_write_count
);

    localparam NUM_GROUPS      = 2;
    localparam GROUP_ID_WIDTH  = 1;
    localparam GLOBAL_ID_WIDTH = GROUP_ID_WIDTH + CORE_ID_WIDTH;
    localparam MAX_FANOUT      = 16;
    localparam FANOUT_IDX_W    = 4;
    localparam CT_ENTRIES      = INPUT_NEURONS * OUTPUT_NEURONS;
    localparam [15:0] INPUT_THRESHOLD_VALUE = INPUT_THRESHOLD;
    localparam [15:0] OUTPUT_THRESHOLD_VALUE = OUTPUT_THRESHOLD;
    localparam [CORE_ID_WIDTH-1:0] OUTPUT_COUNT_ID = OUTPUT_NEURONS;

    localparam [3:0]
        ST_INIT_CT     = 4'd0,
        ST_INIT_FLUSH  = 4'd1,
        ST_RUN         = 4'd2,
        ST_WAIT_DRAIN  = 4'd3,
        ST_LEARN       = 4'd4,
        ST_LEARN_FLUSH = 4'd5,
        ST_DONE        = 4'd6;

    localparam [WEIGHT_WIDTH-1:0] W_MIN_VALUE = W_MIN;
    localparam [WEIGHT_WIDTH-1:0] W_MAX_VALUE = W_MAX;

    reg [3:0] state;
    reg [TIME_WIDTH-1:0] current_time;

    always @(posedge clk) begin
        if (!rst_n)
            current_time <= {TIME_WIDTH{1'b0}};
        else if (enable)
            current_time <= current_time + {{(TIME_WIDTH-1){1'b0}}, 1'b1};
    end

    //-------------------------------------------------------------------------
    // Router, CT, and two real core_group instances.
    //-------------------------------------------------------------------------
    wire [NUM_GROUPS-1:0] router_grp_spike_valid;
    wire [NUM_GROUPS*CORE_ID_WIDTH-1:0] router_grp_spike_neuron_id;
    wire [NUM_GROUPS-1:0] router_grp_spike_ready;
    wire [NUM_GROUPS-1:0] router_grp_in_valid;
    wire [NUM_GROUPS*CORE_ID_WIDTH-1:0] router_grp_in_dest_id;
    wire [NUM_GROUPS*WEIGHT_WIDTH-1:0] router_grp_in_weight;
    wire [NUM_GROUPS-1:0] router_grp_in_exc;
    wire [NUM_GROUPS-1:0] router_grp_in_ready;
    wire [NUM_GROUPS-1:0] router_grp_weight_we;
    wire [CORE_ID_WIDTH-1:0] router_grp_weight_src;
    wire [CORE_ID_WIDTH-1:0] router_grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0] router_grp_weight_data;
    wire router_grp_weight_exc;

    wire cg0_ext_ready;
    wire cg0_out_valid;
    wire [CORE_ID_WIDTH-1:0] cg0_out_id;
    wire [15:0] cg0_spike_count;
    wire cg0_busy;

    wire cg1_ext_ready;
    wire cg1_out_valid;
    wire [CORE_ID_WIDTH-1:0] cg1_out_id;
    wire [15:0] cg1_spike_count;
    wire cg1_busy;

    reg group0_spike_pending;
    reg [CORE_ID_WIDTH-1:0] group0_spike_id;
    reg group1_spike_pending;
    reg [CORE_ID_WIDTH-1:0] group1_spike_id;

    wire drain_active = (state == ST_WAIT_DRAIN);
    wire group0_output_capture = cg0_out_valid && !group0_spike_pending;
    wire group1_output_capture = cg1_out_valid && !group1_spike_pending;

    assign router_grp_spike_valid = {group1_spike_pending, group0_spike_pending};
    assign router_grp_spike_neuron_id = {group1_spike_id, group0_spike_id};
    assign router_grp_in_ready = {cg1_ext_ready, cg0_ext_ready};

    core_group #(
        .GROUP_ID(0),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(16),
        .LEAK_WIDTH(8),
        .REFRAC_WIDTH(8),
        .SPIKE_BUFFER_DEPTH(64)
    ) u_input_group (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .ext_spike_valid(router_grp_in_valid[0]),
        .ext_spike_dest_id(router_grp_in_dest_id[CORE_ID_WIDTH-1:0]),
        .ext_spike_weight(router_grp_in_weight[WEIGHT_WIDTH-1:0]),
        .ext_spike_exc_inh(router_grp_in_exc[0]),
        .ext_spike_ready(cg0_ext_ready),
        .out_spike_valid(cg0_out_valid),
        .out_spike_neuron_id(cg0_out_id),
        .out_spike_ready(!group0_spike_pending),
        .global_threshold(INPUT_THRESHOLD_VALUE),
        .global_leak_rate(drain_active ? 8'd1 : 8'd0),
        .global_refrac_period(8'd8),
        .weight_we(router_grp_weight_we[0]),
        .weight_src_id(router_grp_weight_src),
        .weight_dst_id(router_grp_weight_dst),
        .weight_data(router_grp_weight_data),
        .weight_exc(router_grp_weight_exc),
        .spike_count(cg0_spike_count),
        .group_busy(cg0_busy)
    );

    core_group #(
        .GROUP_ID(1),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(16),
        .LEAK_WIDTH(8),
        .REFRAC_WIDTH(8),
        .SPIKE_BUFFER_DEPTH(64)
    ) u_output_group (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .ext_spike_valid(router_grp_in_valid[1]),
        .ext_spike_dest_id(router_grp_in_dest_id[CORE_ID_WIDTH +: CORE_ID_WIDTH]),
        .ext_spike_weight(router_grp_in_weight[WEIGHT_WIDTH +: WEIGHT_WIDTH]),
        .ext_spike_exc_inh(router_grp_in_exc[1]),
        .ext_spike_ready(cg1_ext_ready),
        .out_spike_valid(cg1_out_valid),
        .out_spike_neuron_id(cg1_out_id),
        .out_spike_ready(!group1_spike_pending),
        .global_threshold(OUTPUT_THRESHOLD_VALUE),
        .global_leak_rate(drain_active ? 8'd1 : 8'd0),
        .global_refrac_period(8'd8),
        .weight_we(router_grp_weight_we[1]),
        .weight_src_id(router_grp_weight_src),
        .weight_dst_id(router_grp_weight_dst),
        .weight_data(router_grp_weight_data),
        .weight_exc(router_grp_weight_exc),
        .spike_count(cg1_spike_count),
        .group_busy(cg1_busy)
    );

    //-------------------------------------------------------------------------
    // Router external input staging.
    //-------------------------------------------------------------------------
    reg router_event_pending;
    reg router_event_issued;
    reg [CORE_ID_WIDTH-1:0] router_event_dest;
    wire router_ext_ready;
    wire router_busy;
    wire router_ext_valid;

    assign event_ready = (state == ST_RUN) && init_done && !router_event_pending;
    assign router_ext_valid = router_event_pending && !router_event_issued && router_ext_ready;

    //-------------------------------------------------------------------------
    // CT setup/learning request staging through event_router_ng.learn_weight_*.
    //-------------------------------------------------------------------------
    reg [INPUT_ID_WIDTH-1:0] init_src;
    reg [OUTPUT_ID_WIDTH-1:0] init_out;
    reg [INPUT_ID_WIDTH-1:0] learn_src;
    reg [OUTPUT_ID_WIDTH-1:0] learned_winner;
    reg [3:0] flush_count;

    reg [WEIGHT_WIDTH-1:0] ct_weight_shadow [0:OUTPUT_NEURONS-1][0:INPUT_NEURONS-1];
    reg [EVENT_WEIGHT_WIDTH:0] active_trace [0:INPUT_NEURONS-1];

    function [WEIGHT_WIDTH-1:0] initial_weight;
        input [OUTPUT_ID_WIDTH-1:0] out_idx;
        input [INPUT_ID_WIDTH-1:0] in_idx;
        integer pattern;
    begin
        pattern = (out_idx * 13 + in_idx * 7 + (in_idx >> 3) * 5 + (in_idx & 7) * 3) % 10;
        initial_weight = 4 + pattern;
    end
    endfunction

    function [WEIGHT_WIDTH-1:0] next_active_weight;
        input [OUTPUT_ID_WIDTH-1:0] out_idx;
        input [INPUT_ID_WIDTH-1:0] in_idx;
        integer candidate;
    begin
        candidate = ct_weight_shadow[out_idx][in_idx] + A_PLUS + (active_trace[in_idx] >> 3);
        if (candidate > W_MAX)
            candidate = W_MAX;
        next_active_weight = candidate[WEIGHT_WIDTH-1:0];
    end
    endfunction

    wire router_learn_weight_ready;
    wire init_request = (state == ST_INIT_CT) && router_learn_weight_ready;
    wire learn_src_active = (active_trace[learn_src] != 0);
    wire learn_request = (state == ST_LEARN) && learn_src_active && router_learn_weight_ready;
    wire [WEIGHT_WIDTH-1:0] learned_next_weight = next_active_weight(learned_winner, learn_src);

    wire router_learn_weight_valid = init_request || learn_request;
    wire [CORE_ID_WIDTH-1:0] router_learn_weight_src =
        init_request ? {1'b0, init_src} : {1'b0, learn_src};
    wire [CORE_ID_WIDTH-1:0] router_learn_weight_dst =
        init_request ? {3'b000, init_out} : {3'b000, learned_winner};
    wire [WEIGHT_WIDTH-1:0] router_learn_weight_data =
        init_request ? ct_weight_shadow[init_out][init_src] : learned_next_weight;
    wire [FANOUT_IDX_W-1:0] router_learn_weight_fanout =
        init_request ? init_out[FANOUT_IDX_W-1:0] : learned_winner[FANOUT_IDX_W-1:0];

    wire router_learn_spike_valid;
    wire [GLOBAL_ID_WIDTH-1:0] router_learn_spike_src_id;

    wire ct_lookup_en;
    wire [GROUP_ID_WIDTH-1:0] ct_lookup_src_group;
    wire [CORE_ID_WIDTH-1:0] ct_lookup_src_neuron;
    wire [FANOUT_IDX_W-1:0] ct_lookup_fanout_idx;
    wire ct_result_valid;
    wire [GROUP_ID_WIDTH-1:0] ct_result_dst_group;
    wire [CORE_ID_WIDTH-1:0] ct_result_dst_neuron;
    wire [WEIGHT_WIDTH-1:0] ct_result_weight;
    wire ct_result_exc_inh;
    wire ct_result_entry_valid;
    wire ct_cfg_we;
    wire [GROUP_ID_WIDTH-1:0] ct_cfg_src_group;
    wire [CORE_ID_WIDTH-1:0] ct_cfg_src_neuron;
    wire [FANOUT_IDX_W-1:0] ct_cfg_fanout_idx;
    wire ct_cfg_valid;
    wire [GROUP_ID_WIDTH-1:0] ct_cfg_dst_group;
    wire [CORE_ID_WIDTH-1:0] ct_cfg_dst_neuron;
    wire [WEIGHT_WIDTH-1:0] ct_cfg_weight;
    wire ct_cfg_exc_inh;

    event_router_ng #(
        .NUM_GROUPS(NUM_GROUPS),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_FANOUT_INTER(MAX_FANOUT),
        .GROUP_ID_WIDTH(GROUP_ID_WIDTH),
        .LOCAL_ID_WIDTH(CORE_ID_WIDTH),
        .GLOBAL_ID_WIDTH(GLOBAL_ID_WIDTH),
        .FANOUT_IDX_WIDTH(FANOUT_IDX_W)
    ) u_event_router (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .grp_spike_valid(router_grp_spike_valid),
        .grp_spike_neuron_id(router_grp_spike_neuron_id),
        .grp_spike_ready(router_grp_spike_ready),
        .grp_in_valid(router_grp_in_valid),
        .grp_in_dest_id(router_grp_in_dest_id),
        .grp_in_weight(router_grp_in_weight),
        .grp_in_exc(router_grp_in_exc),
        .grp_in_ready(router_grp_in_ready),
        .ext_spike_valid(router_ext_valid),
        .ext_spike_neuron_id({1'b0, router_event_dest}),
        .ext_spike_weight(8'd15),
        .ext_spike_exc(1'b1),
        .ext_spike_ready(router_ext_ready),
        .learn_spike_valid(router_learn_spike_valid),
        .learn_spike_src_id(router_learn_spike_src_id),
        .learn_spike_ready(1'b1),
        .learn_weight_valid(router_learn_weight_valid),
        .learn_weight_group(1'b0),
        .learn_weight_src(router_learn_weight_src),
        .learn_weight_dst(router_learn_weight_dst),
        .learn_weight_data(router_learn_weight_data),
        .learn_weight_exc(1'b1),
        .learn_weight_is_inter(1'b1),
        .learn_weight_dst_group(1'b1),
        .learn_weight_fanout_idx(router_learn_weight_fanout),
        .learn_weight_ready(router_learn_weight_ready),
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
        .grp_weight_we(router_grp_weight_we),
        .grp_weight_src(router_grp_weight_src),
        .grp_weight_dst(router_grp_weight_dst),
        .grp_weight_data(router_grp_weight_data),
        .grp_weight_exc(router_grp_weight_exc),
        .ct_cfg_we(ct_cfg_we),
        .ct_cfg_src_group(ct_cfg_src_group),
        .ct_cfg_src_neuron(ct_cfg_src_neuron),
        .ct_cfg_fanout_idx(ct_cfg_fanout_idx),
        .ct_cfg_valid(ct_cfg_valid),
        .ct_cfg_dst_group(ct_cfg_dst_group),
        .ct_cfg_dst_neuron(ct_cfg_dst_neuron),
        .ct_cfg_weight(ct_cfg_weight),
        .ct_cfg_exc_inh(ct_cfg_exc_inh),
        .routed_spike_count(router_routed_spike_count),
        .router_busy(router_busy)
    );

    synaptic_connectivity_table #(
        .NUM_GROUPS(NUM_GROUPS),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_FANOUT_INTER(MAX_FANOUT),
        .GROUP_ID_WIDTH(GROUP_ID_WIDTH),
        .LOCAL_ID_WIDTH(CORE_ID_WIDTH),
        .GLOBAL_ID_WIDTH(GLOBAL_ID_WIDTH),
        .FANOUT_IDX_WIDTH(FANOUT_IDX_W)
    ) u_connectivity_table (
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

    assign direct_learning_write_count = 32'd0;
    assign direct_ct_write_count = 32'd0;

    //-------------------------------------------------------------------------
    // WTA observes natural group-1 output spikes.
    //-------------------------------------------------------------------------
    wire group1_classifier_spike =
        group1_output_capture &&
        (cg1_out_id < OUTPUT_COUNT_ID);
    wire [OUTPUT_ID_WIDTH-1:0] group1_output_id = cg1_out_id[OUTPUT_ID_WIDTH-1:0];

    wire wta_winner_valid;
    wire [OUTPUT_ID_WIDTH-1:0] wta_winner_id;
    wire wta_inhibit_active;

    winner_take_all #(
        .NUM_OUTPUTS(OUTPUT_NEURONS),
        .OUTPUT_ID_WIDTH(OUTPUT_ID_WIDTH),
        .INHIBIT_CYCLES(WTA_INHIBIT_CYCLES),
        .INHIBIT_CNT_WIDTH(16)
    ) u_wta (
        .clk(clk),
        .rst_n(rst_n),
        .enable(state == ST_RUN || state == ST_WAIT_DRAIN),
        .spike_valid(group1_classifier_spike),
        .spike_neuron_id(group1_output_id),
        .winner_valid(wta_winner_valid),
        .winner_neuron_id(wta_winner_id),
        .inhibit_active(wta_inhibit_active)
    );

    //-------------------------------------------------------------------------
    // Bookkeeping.
    //-------------------------------------------------------------------------
    reg [TIME_WIDTH-1:0] image_start_time;
    reg [15:0] drain_count;
    reg winner_seen;
    reg [TIME_WIDTH-1:0] first_winner_latency;
    reg [15:0] current_image_output_spikes;
    reg image_end_requested;

    integer out_i;
    integer in_i;
    integer metric_sum;
    integer metric_min;
    integer metric_max;

    task automatic update_weight_metrics;
    begin
        metric_sum = 0;
        metric_min = W_MAX;
        metric_max = W_MIN;
        for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
            for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1) begin
                metric_sum = metric_sum + ct_weight_shadow[out_i][in_i];
                if (ct_weight_shadow[out_i][in_i] < metric_min)
                    metric_min = ct_weight_shadow[out_i][in_i];
                if (ct_weight_shadow[out_i][in_i] > metric_max)
                    metric_max = ct_weight_shadow[out_i][in_i];
            end
        end
        debug_weight_sum <= metric_sum[31:0];
        debug_weight_min <= metric_min[WEIGHT_WIDTH-1:0];
        debug_weight_max <= metric_max[WEIGHT_WIDTH-1:0];
    end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state                       <= ST_INIT_CT;
            init_done                   <= 1'b0;
            image_done                  <= 1'b0;
            winner_valid                <= 1'b0;
            winner_id                   <= {OUTPUT_ID_WIDTH{1'b0}};
            latency_cycles              <= {TIME_WIDTH{1'b0}};
            image_output_spike_count    <= 16'd0;
            total_images                <= 32'd0;
            total_input_spikes          <= 32'd0;
            total_group0_spikes         <= 32'd0;
            total_group1_spikes         <= 32'd0;
            total_ct_changed_weights    <= 32'd0;
            debug_weight_min            <= W_MAX_VALUE;
            debug_weight_max            <= W_MIN_VALUE;
            debug_weight_sum            <= 32'd0;
            router_observed_spike_count <= 32'd0;
            inter_group_routed_spikes   <= 32'd0;
            ct_init_write_count         <= 32'd0;
            ct_learned_update_count     <= 32'd0;
            router_event_pending        <= 1'b0;
            router_event_issued         <= 1'b0;
            router_event_dest           <= {CORE_ID_WIDTH{1'b0}};
            group0_spike_pending        <= 1'b0;
            group0_spike_id             <= {CORE_ID_WIDTH{1'b0}};
            group1_spike_pending        <= 1'b0;
            group1_spike_id             <= {CORE_ID_WIDTH{1'b0}};
            init_src                    <= {INPUT_ID_WIDTH{1'b0}};
            init_out                    <= {OUTPUT_ID_WIDTH{1'b0}};
            learn_src                   <= {INPUT_ID_WIDTH{1'b0}};
            learned_winner              <= {OUTPUT_ID_WIDTH{1'b0}};
            flush_count                 <= 4'd0;
            image_start_time            <= {TIME_WIDTH{1'b0}};
            drain_count                 <= 16'd0;
            winner_seen                 <= 1'b0;
            first_winner_latency        <= {TIME_WIDTH{1'b0}};
            current_image_output_spikes <= 16'd0;
            image_end_requested         <= 1'b0;

            for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                    ct_weight_shadow[out_i][in_i] <= initial_weight(out_i[OUTPUT_ID_WIDTH-1:0], in_i[INPUT_ID_WIDTH-1:0]);
            end
            for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                active_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
        end else begin
            image_done   <= 1'b0;
            winner_valid <= 1'b0;

            if (router_learn_spike_valid)
                router_observed_spike_count <= router_observed_spike_count + 32'd1;

            if (router_grp_in_valid[1])
                inter_group_routed_spikes <= inter_group_routed_spikes + 32'd1;

            if (group0_spike_pending && router_grp_spike_ready[0])
                group0_spike_pending <= 1'b0;
            else if (group0_output_capture) begin
                group0_spike_pending <= 1'b1;
                group0_spike_id      <= cg0_out_id;
            end

            if (group1_spike_pending && router_grp_spike_ready[1])
                group1_spike_pending <= 1'b0;
            else if (group1_output_capture) begin
                group1_spike_pending <= 1'b1;
                group1_spike_id      <= cg1_out_id;
            end

            if (group0_output_capture)
                total_group0_spikes <= total_group0_spikes + 32'd1;
            if (group1_output_capture)
                total_group1_spikes <= total_group1_spikes + 32'd1;

            if (router_event_pending && !router_event_issued && router_ext_ready)
                router_event_issued <= 1'b1;
            if (router_grp_in_valid[0]) begin
                router_event_pending <= 1'b0;
                router_event_issued  <= 1'b0;
            end

            if (group1_classifier_spike)
                current_image_output_spikes <= current_image_output_spikes + 16'd1;

            if (wta_winner_valid && !winner_seen) begin
                winner_seen          <= 1'b1;
                learned_winner       <= wta_winner_id;
                first_winner_latency <= current_time - image_start_time;
            end

            case (state)
                ST_INIT_CT: begin
                    if (init_request) begin
                        ct_init_write_count <= ct_init_write_count + 32'd1;
                        if (init_src == INPUT_NEURONS-1) begin
                            init_src <= {INPUT_ID_WIDTH{1'b0}};
                            if (init_out == OUTPUT_NEURONS-1) begin
                                init_out    <= {OUTPUT_ID_WIDTH{1'b0}};
                                flush_count <= 4'd0;
                                state       <= ST_INIT_FLUSH;
                            end else begin
                                init_out <= init_out + 1'b1;
                            end
                        end else begin
                            init_src <= init_src + 1'b1;
                        end
                    end
                end

                ST_INIT_FLUSH: begin
                    if (flush_count >= 4'd8) begin
                        init_done <= 1'b1;
                        state     <= ST_RUN;
                        update_weight_metrics;
                    end else begin
                        flush_count <= flush_count + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (image_start) begin
                        image_start_time            <= current_time;
                        winner_seen                 <= 1'b0;
                        first_winner_latency        <= {TIME_WIDTH{1'b0}};
                        current_image_output_spikes <= 16'd0;
                        image_end_requested         <= 1'b0;
                        for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                            active_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
                    end

                    if (event_valid && event_ready) begin
                        router_event_pending <= 1'b1;
                        router_event_issued  <= 1'b0;
                        router_event_dest    <= {1'b0, event_neuron_id};
                        total_input_spikes   <= total_input_spikes + 32'd1;
                        if (active_trace[event_neuron_id] + event_weight > 15)
                            active_trace[event_neuron_id] <= 5'd15;
                        else
                            active_trace[event_neuron_id] <= active_trace[event_neuron_id] + event_weight;
                    end

                    if (image_end)
                        image_end_requested <= 1'b1;

                    if (image_end_requested &&
                        !router_event_pending &&
                        !router_busy &&
                        !group0_spike_pending &&
                        !group1_spike_pending) begin
                        drain_count <= 16'd0;
                        state       <= ST_WAIT_DRAIN;
                    end
                end

                ST_WAIT_DRAIN: begin
                    if (drain_count >= DRAIN_CYCLES) begin
                        if (learning_enable && winner_seen) begin
                            learn_src <= {INPUT_ID_WIDTH{1'b0}};
                            state     <= ST_LEARN;
                        end else begin
                            state <= ST_DONE;
                        end
                    end else begin
                        drain_count <= drain_count + 16'd1;
                    end
                end

                ST_LEARN: begin
                    if (!learn_src_active) begin
                        if (learn_src == INPUT_NEURONS-1) begin
                            flush_count <= 4'd0;
                            state       <= ST_LEARN_FLUSH;
                        end else begin
                            learn_src <= learn_src + 1'b1;
                        end
                    end else if (learn_request) begin
                        if (learned_next_weight != ct_weight_shadow[learned_winner][learn_src]) begin
                            total_ct_changed_weights <= total_ct_changed_weights + 32'd1;
                            ct_learned_update_count  <= ct_learned_update_count + 32'd1;
                        end
                        ct_weight_shadow[learned_winner][learn_src] <= learned_next_weight;

                        if (learn_src == INPUT_NEURONS-1) begin
                            flush_count <= 4'd0;
                            state       <= ST_LEARN_FLUSH;
                        end else begin
                            learn_src <= learn_src + 1'b1;
                        end
                    end
                end

                ST_LEARN_FLUSH: begin
                    if (flush_count >= 4'd8) begin
                        update_weight_metrics;
                        state <= ST_DONE;
                    end else begin
                        flush_count <= flush_count + 1'b1;
                    end
                end

                ST_DONE: begin
                    image_done               <= 1'b1;
                    winner_valid             <= winner_seen;
                    winner_id                <= learned_winner;
                    latency_cycles           <= winner_seen ? first_winner_latency : (current_time - image_start_time);
                    image_output_spike_count <= current_image_output_spikes;
                    total_images             <= total_images + 32'd1;
                    state                    <= ST_RUN;
                end

                default: state <= ST_INIT_CT;
            endcase
        end
    end

endmodule
