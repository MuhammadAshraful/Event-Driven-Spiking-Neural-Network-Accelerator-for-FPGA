//=============================================================================
// Router-wrapped one-core_group MNIST classifier top
//
// This is the next staged path after custom_rtl_mnist_coregroup_classifier_top:
//   - MNIST input spikes enter through event_router_ng external-spike input
//   - event_router_ng delivers those spikes to one real core_group
//   - output spikes are natural core_group/LIF spikes
//   - STDP/WTA learning updates are issued through event_router_ng.learn_weight_*
//   - event_router_ng forwards learned intra-group weights to core_group.weight_*
//
// NUM_GROUPS is set to 2 because a true NUM_GROUPS=1 instance creates a
// zero-width group id in event_router_ng. Only group 0 is active here.
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_rtl_mnist_router_coregroup_classifier_top #(
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
    parameter A_MINUS            = 1,
    parameter OUTPUT_BASE        = 64,
    parameter LIF_THRESHOLD      = 8,
    parameter DRAIN_CYCLES       = 1200,
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
    output reg  [31:0]                   total_output_spikes,
    output reg  [31:0]                   total_weight_updates,
    output reg  [31:0]                   total_input_spikes,
    output reg  [WEIGHT_WIDTH-1:0]       debug_weight_min,
    output reg  [WEIGHT_WIDTH-1:0]       debug_weight_max,
    output reg  [31:0]                   debug_weight_sum,

    output wire [31:0]                   router_routed_spike_count,
    output reg  [31:0]                   router_observed_spike_count,
    output reg  [31:0]                   router_learning_weight_writes,
    output wire [31:0]                   direct_learning_write_count
);

    localparam ROUTER_GROUPS     = 2;
    localparam ROUTER_GROUP_W    = 1;
    localparam ROUTER_GLOBAL_W   = ROUTER_GROUP_W + CORE_ID_WIDTH;
    localparam ROUTER_FANOUT_MAX = 4;
    localparam ROUTER_FANOUT_W   = 2;

    localparam [3:0]
        ST_INIT       = 4'd0,
        ST_INIT_FLUSH = 4'd1,
        ST_RUN        = 4'd2,
        ST_WAIT_DRAIN = 4'd3,
        ST_LEARN      = 4'd4,
        ST_LEARN_FLUSH= 4'd5,
        ST_DONE       = 4'd6;

    localparam [WEIGHT_WIDTH-1:0] W_MIN_VALUE = W_MIN;
    localparam [WEIGHT_WIDTH-1:0] W_MAX_VALUE = W_MAX;
    localparam [CORE_ID_WIDTH-1:0] OUTPUT_BASE_ID = OUTPUT_BASE;
    localparam [CORE_ID_WIDTH-1:0] OUTPUT_LIMIT_ID = OUTPUT_BASE + OUTPUT_NEURONS;
    localparam [15:0] LIF_THRESHOLD_VALUE = LIF_THRESHOLD;

    reg [3:0] state;
    reg [TIME_WIDTH-1:0] current_time;

    always @(posedge clk) begin
        if (!rst_n)
            current_time <= {TIME_WIDTH{1'b0}};
        else if (enable)
            current_time <= current_time + {{(TIME_WIDTH-1){1'b0}}, 1'b1};
    end

    //-------------------------------------------------------------------------
    // One real core_group. All spike processing and local recurrent fanout use
    // core_group's own LIF state and dense local weight memory.
    //-------------------------------------------------------------------------
    wire                        cg_ext_ready;
    wire                        cg_out_valid;
    wire [CORE_ID_WIDTH-1:0]    cg_out_id;
    wire [15:0]                 cg_spike_count;
    wire                        cg_busy;

    wire [ROUTER_GROUPS-1:0] router_grp_spike_valid;
    wire [ROUTER_GROUPS*CORE_ID_WIDTH-1:0] router_grp_spike_neuron_id;
    wire [ROUTER_GROUPS-1:0] router_grp_spike_ready;
    wire [ROUTER_GROUPS-1:0] router_grp_in_valid;
    wire [ROUTER_GROUPS*CORE_ID_WIDTH-1:0] router_grp_in_dest_id;
    wire [ROUTER_GROUPS*WEIGHT_WIDTH-1:0] router_grp_in_weight;
    wire [ROUTER_GROUPS-1:0] router_grp_in_exc;
    wire [ROUTER_GROUPS-1:0] router_grp_in_ready;
    wire [ROUTER_GROUPS-1:0] router_grp_weight_we;
    wire [CORE_ID_WIDTH-1:0] router_grp_weight_src;
    wire [CORE_ID_WIDTH-1:0] router_grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0] router_grp_weight_data;
    wire router_grp_weight_exc;
    reg router_group_spike_pending;
    reg [CORE_ID_WIDTH-1:0] router_group_spike_id;

    wire drain_active = (state == ST_WAIT_DRAIN);

    wire core_output_capture = cg_out_valid && !router_group_spike_pending;

    assign router_grp_spike_valid = {1'b0, router_group_spike_pending};
    assign router_grp_spike_neuron_id = {{CORE_ID_WIDTH{1'b0}}, router_group_spike_id};
    assign router_grp_in_ready = {1'b1, cg_ext_ready};

    core_group #(
        .GROUP_ID(0),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(16),
        .LEAK_WIDTH(8),
        .REFRAC_WIDTH(8),
        .SPIKE_BUFFER_DEPTH(64)
    ) u_core_group (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .ext_spike_valid(router_grp_in_valid[0]),
        .ext_spike_dest_id(router_grp_in_dest_id[CORE_ID_WIDTH-1:0]),
        .ext_spike_weight(router_grp_in_weight[WEIGHT_WIDTH-1:0]),
        .ext_spike_exc_inh(router_grp_in_exc[0]),
        .ext_spike_ready(cg_ext_ready),
        .out_spike_valid(cg_out_valid),
        .out_spike_neuron_id(cg_out_id),
        .out_spike_ready(!router_group_spike_pending),
        .global_threshold(LIF_THRESHOLD_VALUE),
        .global_leak_rate(drain_active ? 8'd1 : 8'd0),
        .global_refrac_period(8'd8),
        .weight_we(router_grp_weight_we[0]),
        .weight_src_id(router_grp_weight_src),
        .weight_dst_id(router_grp_weight_dst),
        .weight_data(router_grp_weight_data),
        .weight_exc(router_grp_weight_exc),
        .spike_count(cg_spike_count),
        .group_busy(cg_busy)
    );

    //-------------------------------------------------------------------------
    // Router external input and learned-weight request staging.
    //-------------------------------------------------------------------------
    reg router_event_pending;
    reg router_event_issued;
    reg [CORE_ID_WIDTH-1:0] router_event_dest;
    reg [WEIGHT_WIDTH-1:0] router_event_weight;
    wire router_ext_ready;
    wire router_busy;
    wire router_ext_valid;

    assign event_ready = (state == ST_RUN) && init_done && !router_event_pending;
    assign router_ext_valid = router_event_pending && !router_event_issued && router_ext_ready;

    reg [CORE_ID_WIDTH-1:0] init_src;
    reg [OUTPUT_ID_WIDTH-1:0] init_out;
    reg [INPUT_ID_WIDTH-1:0] learn_src;
    reg [OUTPUT_ID_WIDTH-1:0] learned_winner;
    reg [3:0] flush_count;

    reg [WEIGHT_WIDTH-1:0] weight_shadow [0:OUTPUT_NEURONS-1][0:INPUT_NEURONS-1];
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

    function [WEIGHT_WIDTH-1:0] next_learned_weight;
        input [OUTPUT_ID_WIDTH-1:0] out_idx;
        input [INPUT_ID_WIDTH-1:0] in_idx;
        integer candidate;
    begin
        if (active_trace[in_idx] != 0) begin
            candidate = weight_shadow[out_idx][in_idx] + A_PLUS + (active_trace[in_idx] >> 3);
            if (candidate > W_MAX)
                candidate = W_MAX;
        end else begin
            candidate = weight_shadow[out_idx][in_idx] - A_MINUS;
            if (candidate < W_MIN)
                candidate = W_MIN;
        end
        next_learned_weight = candidate[WEIGHT_WIDTH-1:0];
    end
    endfunction

    wire router_learn_weight_ready;
    wire issue_init_weight = (state == ST_INIT) && router_learn_weight_ready;
    wire issue_learn_weight = (state == ST_LEARN) && router_learn_weight_ready;
    wire [WEIGHT_WIDTH-1:0] learn_next_weight = next_learned_weight(learned_winner, learn_src);

    wire router_learn_weight_valid = issue_init_weight || issue_learn_weight;
    wire [CORE_ID_WIDTH-1:0] router_learn_weight_src =
        issue_init_weight ? init_src : {1'b0, learn_src};
    wire [CORE_ID_WIDTH-1:0] router_learn_weight_dst =
        issue_init_weight ? (OUTPUT_BASE_ID + init_out) : (OUTPUT_BASE_ID + learned_winner);
    wire [WEIGHT_WIDTH-1:0] router_learn_weight_data =
        issue_init_weight ? weight_shadow[init_out][init_src[INPUT_ID_WIDTH-1:0]] : learn_next_weight;
    wire router_learn_spike_valid;
    wire [ROUTER_GLOBAL_W-1:0] router_learn_spike_src_id;

    wire router_ct_lookup_en;
    wire [ROUTER_GROUP_W-1:0] router_ct_lookup_src_group;
    wire [CORE_ID_WIDTH-1:0] router_ct_lookup_src_neuron;
    wire [ROUTER_FANOUT_W-1:0] router_ct_lookup_fanout_idx;
    wire router_ct_cfg_we;
    wire [ROUTER_GROUP_W-1:0] router_ct_cfg_src_group;
    wire [CORE_ID_WIDTH-1:0] router_ct_cfg_src_neuron;
    wire [ROUTER_FANOUT_W-1:0] router_ct_cfg_fanout_idx;
    wire router_ct_cfg_valid;
    wire [ROUTER_GROUP_W-1:0] router_ct_cfg_dst_group;
    wire [CORE_ID_WIDTH-1:0] router_ct_cfg_dst_neuron;
    wire [WEIGHT_WIDTH-1:0] router_ct_cfg_weight;
    wire router_ct_cfg_exc_inh;

    event_router_ng #(
        .NUM_GROUPS(ROUTER_GROUPS),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_FANOUT_INTER(ROUTER_FANOUT_MAX),
        .GROUP_ID_WIDTH(ROUTER_GROUP_W),
        .LOCAL_ID_WIDTH(CORE_ID_WIDTH),
        .GLOBAL_ID_WIDTH(ROUTER_GLOBAL_W),
        .FANOUT_IDX_WIDTH(ROUTER_FANOUT_W)
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
        .ext_spike_weight(router_event_weight),
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
        .learn_weight_is_inter(1'b0),
        .learn_weight_dst_group(1'b0),
        .learn_weight_fanout_idx({ROUTER_FANOUT_W{1'b0}}),
        .learn_weight_ready(router_learn_weight_ready),
        .ct_lookup_en(router_ct_lookup_en),
        .ct_lookup_src_group(router_ct_lookup_src_group),
        .ct_lookup_src_neuron(router_ct_lookup_src_neuron),
        .ct_lookup_fanout_idx(router_ct_lookup_fanout_idx),
        .ct_result_valid(1'b0),
        .ct_result_dst_group(1'b0),
        .ct_result_dst_neuron({CORE_ID_WIDTH{1'b0}}),
        .ct_result_weight({WEIGHT_WIDTH{1'b0}}),
        .ct_result_exc_inh(1'b1),
        .ct_result_entry_valid(1'b0),
        .grp_weight_we(router_grp_weight_we),
        .grp_weight_src(router_grp_weight_src),
        .grp_weight_dst(router_grp_weight_dst),
        .grp_weight_data(router_grp_weight_data),
        .grp_weight_exc(router_grp_weight_exc),
        .ct_cfg_we(router_ct_cfg_we),
        .ct_cfg_src_group(router_ct_cfg_src_group),
        .ct_cfg_src_neuron(router_ct_cfg_src_neuron),
        .ct_cfg_fanout_idx(router_ct_cfg_fanout_idx),
        .ct_cfg_valid(router_ct_cfg_valid),
        .ct_cfg_dst_group(router_ct_cfg_dst_group),
        .ct_cfg_dst_neuron(router_ct_cfg_dst_neuron),
        .ct_cfg_weight(router_ct_cfg_weight),
        .ct_cfg_exc_inh(router_ct_cfg_exc_inh),
        .routed_spike_count(router_routed_spike_count),
        .router_busy(router_busy)
    );

    assign direct_learning_write_count = 32'd0;

    //-------------------------------------------------------------------------
    // WTA observes handshaken output spikes from core_group.
    //-------------------------------------------------------------------------
    wire output_spike =
        core_output_capture &&
        (cg_out_id >= OUTPUT_BASE_ID) &&
        (cg_out_id < OUTPUT_LIMIT_ID);
    wire [OUTPUT_ID_WIDTH-1:0] output_local_id =
        cg_out_id - OUTPUT_BASE_ID;

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
        .spike_valid(output_spike),
        .spike_neuron_id(output_local_id),
        .winner_valid(wta_winner_valid),
        .winner_neuron_id(wta_winner_id),
        .inhibit_active(wta_inhibit_active)
    );

    //-------------------------------------------------------------------------
    // Bookkeeping for learning and metrics.
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
                metric_sum = metric_sum + weight_shadow[out_i][in_i];
                if (weight_shadow[out_i][in_i] < metric_min)
                    metric_min = weight_shadow[out_i][in_i];
                if (weight_shadow[out_i][in_i] > metric_max)
                    metric_max = weight_shadow[out_i][in_i];
            end
        end
        debug_weight_sum <= metric_sum[31:0];
        debug_weight_min <= metric_min[WEIGHT_WIDTH-1:0];
        debug_weight_max <= metric_max[WEIGHT_WIDTH-1:0];
    end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state                         <= ST_INIT;
            init_done                     <= 1'b0;
            image_done                    <= 1'b0;
            winner_valid                  <= 1'b0;
            winner_id                     <= {OUTPUT_ID_WIDTH{1'b0}};
            latency_cycles                <= {TIME_WIDTH{1'b0}};
            image_output_spike_count      <= 16'd0;
            total_images                  <= 32'd0;
            total_output_spikes           <= 32'd0;
            total_weight_updates          <= 32'd0;
            total_input_spikes            <= 32'd0;
            debug_weight_min              <= W_MAX_VALUE;
            debug_weight_max              <= W_MIN_VALUE;
            debug_weight_sum              <= 32'd0;
            router_observed_spike_count   <= 32'd0;
            router_learning_weight_writes <= 32'd0;
            router_event_pending          <= 1'b0;
            router_event_issued           <= 1'b0;
            router_event_dest             <= {CORE_ID_WIDTH{1'b0}};
            router_event_weight           <= {WEIGHT_WIDTH{1'b0}};
            router_group_spike_pending    <= 1'b0;
            router_group_spike_id         <= {CORE_ID_WIDTH{1'b0}};
            init_src                      <= {CORE_ID_WIDTH{1'b0}};
            init_out                      <= {OUTPUT_ID_WIDTH{1'b0}};
            learn_src                     <= {INPUT_ID_WIDTH{1'b0}};
            learned_winner                <= {OUTPUT_ID_WIDTH{1'b0}};
            flush_count                   <= 4'd0;
            image_start_time              <= {TIME_WIDTH{1'b0}};
            drain_count                   <= 16'd0;
            winner_seen                   <= 1'b0;
            first_winner_latency          <= {TIME_WIDTH{1'b0}};
            current_image_output_spikes   <= 16'd0;
            image_end_requested           <= 1'b0;

            for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                    weight_shadow[out_i][in_i] <= initial_weight(out_i[OUTPUT_ID_WIDTH-1:0], in_i[INPUT_ID_WIDTH-1:0]);
            end
            for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                active_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
        end else begin
            image_done   <= 1'b0;
            winner_valid <= 1'b0;

            if (router_learn_spike_valid)
                router_observed_spike_count <= router_observed_spike_count + 32'd1;

            if (router_group_spike_pending && router_grp_spike_ready[0])
                router_group_spike_pending <= 1'b0;
            else if (core_output_capture) begin
                router_group_spike_pending <= 1'b1;
                router_group_spike_id      <= cg_out_id;
            end

            if (router_event_pending && !router_event_issued && router_ext_ready)
                router_event_issued <= 1'b1;
            if (router_grp_in_valid[0]) begin
                router_event_pending <= 1'b0;
                router_event_issued  <= 1'b0;
            end

            if (output_spike)
                current_image_output_spikes <= current_image_output_spikes + 16'd1;

            if (wta_winner_valid && !winner_seen) begin
                winner_seen          <= 1'b1;
                learned_winner       <= wta_winner_id;
                first_winner_latency <= current_time - image_start_time;
            end

            case (state)
                ST_INIT: begin
                    if (issue_init_weight) begin
                        if (init_src == INPUT_NEURONS-1) begin
                            init_src <= {CORE_ID_WIDTH{1'b0}};
                            if (init_out + 1 >= OUTPUT_NEURONS) begin
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
                        router_event_weight  <= 8'd15;
                        total_input_spikes   <= total_input_spikes + 32'd1;
                        if (active_trace[event_neuron_id] + event_weight > 15)
                            active_trace[event_neuron_id] <= 5'd15;
                        else
                            active_trace[event_neuron_id] <= active_trace[event_neuron_id] + event_weight;
                    end

                    if (image_end)
                        image_end_requested <= 1'b1;

                    if (image_end_requested && !router_event_pending && !router_busy) begin
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
                    if (issue_learn_weight) begin
                        if (learn_next_weight != weight_shadow[learned_winner][learn_src]) begin
                            total_weight_updates <= total_weight_updates + 32'd1;
                            router_learning_weight_writes <= router_learning_weight_writes + 32'd1;
                        end
                        weight_shadow[learned_winner][learn_src] <= learn_next_weight;

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
                    total_output_spikes      <= total_output_spikes + current_image_output_spikes;
                    state                    <= ST_RUN;
                end

                default: state <= ST_INIT;
            endcase
        end
    end

endmodule
