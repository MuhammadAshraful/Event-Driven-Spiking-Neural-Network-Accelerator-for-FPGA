//=============================================================================
// One-core_group MNIST classifier top
//
// Architecture-faithful classifier path:
//   - one real core_group instance with 128 physical LIF neurons
//   - logical input neurons  : 0..63
//   - logical output neurons : 64..73
//   - input->output synapses live in core_group local weight memory
//   - output spikes are natural core_group/LIF spikes
//   - STDP writes update the real core_group weight memory through weight_we
//
// The small shadow weight array is bookkeeping only because core_group has no
// readback port. It is not used for spike processing or winner selection.
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_rtl_mnist_coregroup_classifier_top #(
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
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    input  wire                         learning_enable,

    input  wire                         image_start,
    input  wire                         event_valid,
    input  wire [INPUT_ID_WIDTH-1:0]    event_neuron_id,
    input  wire [EVENT_WEIGHT_WIDTH-1:0] event_weight,
    input  wire                         image_end,
    output wire                         event_ready,

    output reg                          init_done,
    output reg                          image_done,
    output reg                          winner_valid,
    output reg  [OUTPUT_ID_WIDTH-1:0]   winner_id,
    output reg  [TIME_WIDTH-1:0]        latency_cycles,
    output reg  [15:0]                  image_output_spike_count,

    output reg  [31:0]                  total_images,
    output reg  [31:0]                  total_output_spikes,
    output reg  [31:0]                  total_weight_updates,
    output reg  [31:0]                  total_input_spikes,
    output reg  [WEIGHT_WIDTH-1:0]      debug_weight_min,
    output reg  [WEIGHT_WIDTH-1:0]      debug_weight_max,
    output reg  [31:0]                  debug_weight_sum
);

    localparam [2:0]
        ST_INIT       = 3'd0,
        ST_RUN        = 3'd1,
        ST_WAIT_DRAIN = 3'd2,
        ST_LEARN      = 3'd3,
        ST_DONE       = 3'd4;

    localparam [WEIGHT_WIDTH-1:0] W_MIN_VALUE = W_MIN;
    localparam [WEIGHT_WIDTH-1:0] W_MAX_VALUE = W_MAX;
    localparam [CORE_ID_WIDTH-1:0] OUTPUT_BASE_ID = OUTPUT_BASE;
    localparam [CORE_ID_WIDTH-1:0] OUTPUT_LIMIT_ID = OUTPUT_BASE + OUTPUT_NEURONS;
    localparam [15:0] LIF_THRESHOLD_VALUE = LIF_THRESHOLD;

    reg [2:0] state;
    reg [TIME_WIDTH-1:0] current_time;

    always @(posedge clk) begin
        if (!rst_n)
            current_time <= {TIME_WIDTH{1'b0}};
        else if (enable)
            current_time <= current_time + {{(TIME_WIDTH-1){1'b0}}, 1'b1};
    end

    //-------------------------------------------------------------------------
    // Real core_group / LIF subsystem.
    //-------------------------------------------------------------------------
    reg                         cg_weight_we;
    reg [CORE_ID_WIDTH-1:0]     cg_weight_src;
    reg [CORE_ID_WIDTH-1:0]     cg_weight_dst;
    reg [WEIGHT_WIDTH-1:0]      cg_weight_data;
    reg                         cg_weight_exc;
    wire                        cg_ext_ready;
    wire                        cg_out_valid;
    wire [CORE_ID_WIDTH-1:0]    cg_out_id;
    wire [15:0]                 cg_spike_count;
    wire                        cg_busy;

    wire drain_active = (state == ST_WAIT_DRAIN);

    assign event_ready = (state == ST_RUN) && init_done && cg_ext_ready;

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
        .ext_spike_valid(event_valid && event_ready),
        .ext_spike_dest_id({1'b0, event_neuron_id}),
        .ext_spike_weight(8'd15),
        .ext_spike_exc_inh(1'b1),
        .ext_spike_ready(cg_ext_ready),
        .out_spike_valid(cg_out_valid),
        .out_spike_neuron_id(cg_out_id),
        .out_spike_ready(1'b1),
        .global_threshold(LIF_THRESHOLD_VALUE),
        .global_leak_rate(drain_active ? 8'd1 : 8'd0),
        .global_refrac_period(8'd8),
        .weight_we(cg_weight_we),
        .weight_src_id(cg_weight_src),
        .weight_dst_id(cg_weight_dst),
        .weight_data(cg_weight_data),
        .weight_exc(cg_weight_exc),
        .spike_count(cg_spike_count),
        .group_busy(cg_busy)
    );

    //-------------------------------------------------------------------------
    // WTA observes only output-neuron spikes from core_group.
    //-------------------------------------------------------------------------
    wire output_spike =
        cg_out_valid &&
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
    reg [WEIGHT_WIDTH-1:0] weight_shadow [0:OUTPUT_NEURONS-1][0:INPUT_NEURONS-1];
    reg [EVENT_WEIGHT_WIDTH:0] active_trace [0:INPUT_NEURONS-1];

    reg [CORE_ID_WIDTH-1:0] init_src;
    reg [OUTPUT_ID_WIDTH-1:0] init_out;
    reg [INPUT_ID_WIDTH-1:0] learn_src;
    reg [OUTPUT_ID_WIDTH-1:0] learned_winner;
    reg [TIME_WIDTH-1:0] image_start_time;
    reg [15:0] drain_count;
    reg winner_seen;
    reg [TIME_WIDTH-1:0] first_winner_latency;
    reg [15:0] current_image_output_spikes;

    integer out_i;
    integer in_i;
    integer next_weight;
    integer metric_sum;
    integer metric_min;
    integer metric_max;

    function [WEIGHT_WIDTH-1:0] initial_weight;
        input [OUTPUT_ID_WIDTH-1:0] out_idx;
        input [INPUT_ID_WIDTH-1:0] in_idx;
        integer pattern;
    begin
        pattern = (out_idx * 13 + in_idx * 7 + (in_idx >> 3) * 5 + (in_idx & 7) * 3) % 10;
        initial_weight = 4 + pattern;
    end
    endfunction

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
            state                       <= ST_INIT;
            init_done                   <= 1'b0;
            image_done                  <= 1'b0;
            winner_valid                <= 1'b0;
            winner_id                   <= {OUTPUT_ID_WIDTH{1'b0}};
            latency_cycles              <= {TIME_WIDTH{1'b0}};
            image_output_spike_count    <= 16'd0;
            total_images                <= 32'd0;
            total_output_spikes         <= 32'd0;
            total_weight_updates        <= 32'd0;
            total_input_spikes          <= 32'd0;
            debug_weight_min            <= W_MAX_VALUE;
            debug_weight_max            <= W_MIN_VALUE;
            debug_weight_sum            <= 32'd0;
            cg_weight_we                <= 1'b0;
            cg_weight_src               <= {CORE_ID_WIDTH{1'b0}};
            cg_weight_dst               <= {CORE_ID_WIDTH{1'b0}};
            cg_weight_data              <= {WEIGHT_WIDTH{1'b0}};
            cg_weight_exc               <= 1'b1;
            init_src                    <= {CORE_ID_WIDTH{1'b0}};
            init_out                    <= {OUTPUT_ID_WIDTH{1'b0}};
            learn_src                   <= {INPUT_ID_WIDTH{1'b0}};
            learned_winner              <= {OUTPUT_ID_WIDTH{1'b0}};
            image_start_time            <= {TIME_WIDTH{1'b0}};
            drain_count                 <= 16'd0;
            winner_seen                 <= 1'b0;
            first_winner_latency        <= {TIME_WIDTH{1'b0}};
            current_image_output_spikes <= 16'd0;

            for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                    weight_shadow[out_i][in_i] <= initial_weight(out_i[OUTPUT_ID_WIDTH-1:0], in_i[INPUT_ID_WIDTH-1:0]);
            end
            for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                active_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
        end else begin
            image_done   <= 1'b0;
            winner_valid <= 1'b0;
            cg_weight_we <= 1'b0;

            if (output_spike) begin
                current_image_output_spikes <= current_image_output_spikes + 16'd1;
            end

            if (wta_winner_valid && !winner_seen) begin
                winner_seen          <= 1'b1;
                learned_winner       <= wta_winner_id;
                first_winner_latency <= current_time - image_start_time;
            end

            case (state)
                ST_INIT: begin
                    cg_weight_we   <= 1'b1;
                    cg_weight_src  <= init_src;
                    cg_weight_dst  <= OUTPUT_BASE_ID + init_out;
                    cg_weight_data <= weight_shadow[init_out][init_src[INPUT_ID_WIDTH-1:0]];
                    cg_weight_exc  <= 1'b1;

                    if (init_src + 1 >= INPUT_NEURONS) begin
                        init_src <= {CORE_ID_WIDTH{1'b0}};
                        if (init_out + 1 >= OUTPUT_NEURONS) begin
                            init_out  <= {OUTPUT_ID_WIDTH{1'b0}};
                            init_done <= 1'b1;
                            state     <= ST_RUN;
                            update_weight_metrics;
                        end else begin
                            init_out <= init_out + 1'b1;
                        end
                    end else begin
                        init_src <= init_src + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (image_start) begin
                        image_start_time            <= current_time;
                        winner_seen                 <= 1'b0;
                        first_winner_latency        <= {TIME_WIDTH{1'b0}};
                        current_image_output_spikes <= 16'd0;
                        for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                            active_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
                    end

                    if (event_valid && event_ready) begin
                        total_input_spikes <= total_input_spikes + 32'd1;
                        if (active_trace[event_neuron_id] + event_weight > 15)
                            active_trace[event_neuron_id] <= 5'd15;
                        else
                            active_trace[event_neuron_id] <= active_trace[event_neuron_id] + event_weight;
                    end

                    if (image_end) begin
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
                    if (active_trace[learn_src] != 0) begin
                        next_weight = weight_shadow[learned_winner][learn_src] + A_PLUS + (active_trace[learn_src] >> 3);
                        if (next_weight > W_MAX)
                            next_weight = W_MAX;
                    end else begin
                        next_weight = weight_shadow[learned_winner][learn_src] - A_MINUS;
                        if (next_weight < W_MIN)
                            next_weight = W_MIN;
                    end

                    cg_weight_we   <= 1'b1;
                    cg_weight_src  <= {1'b0, learn_src};
                    cg_weight_dst  <= OUTPUT_BASE_ID + learned_winner;
                    cg_weight_data <= next_weight[WEIGHT_WIDTH-1:0];
                    cg_weight_exc  <= 1'b1;

                    if (next_weight[WEIGHT_WIDTH-1:0] != weight_shadow[learned_winner][learn_src])
                        total_weight_updates <= total_weight_updates + 32'd1;
                    weight_shadow[learned_winner][learn_src] <= next_weight[WEIGHT_WIDTH-1:0];

                    if (learn_src + 1 >= INPUT_NEURONS) begin
                        update_weight_metrics;
                        state <= ST_DONE;
                    end else begin
                        learn_src <= learn_src + 1'b1;
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
