//=============================================================================
// RTL-only unsupervised MNIST classifier top
//
// Practical representative classifier:
//   - 64 input neurons from an 8x8 MNIST image
//   - 10 output neurons
//   - image-window winner-take-all
//   - winner-only STDP-style weight updates
//
// A core_group instance is used as the input spike front-end for accepted input
// events. The classifier matrix is kept directly in this top so xsim can handle
// 10/100 image representative runs without a huge 640-engine STDP fabric.
//=============================================================================

`timescale 1ns / 1ps
`include "snn_params.vh"

module custom_rtl_mnist_classifier_top #(
    parameter INPUT_NEURONS        = 64,
    parameter OUTPUT_NEURONS       = 10,
    parameter CORE_NEURONS         = 128,
    parameter INPUT_ID_WIDTH       = 6,
    parameter OUTPUT_ID_WIDTH      = 4,
    parameter CORE_ID_WIDTH        = 7,
    parameter WEIGHT_WIDTH         = 8,
    parameter EVENT_WEIGHT_WIDTH   = 4,
    parameter MEMBRANE_WIDTH       = 20,
    parameter TIME_WIDTH           = 32,
    parameter W_MIN                = 0,
    parameter W_MAX                = 15,
    parameter A_PLUS               = 2,
    parameter A_MINUS              = 1,
    parameter HOMEOSTASIS_INC      = 16,
    parameter HOMEOSTASIS_DECAY    = 1,
    parameter HOMEOSTASIS_SHIFT    = 2,
    parameter WTA_INHIBIT_CYCLES   = 8
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

    output reg                          image_done,
    output reg                          winner_valid,
    output reg  [OUTPUT_ID_WIDTH-1:0]   winner_id,
    output reg  [TIME_WIDTH-1:0]        latency_cycles,

    output reg  [31:0]                  total_images,
    output reg  [31:0]                  total_output_spikes,
    output reg  [31:0]                  total_weight_updates,
    output reg  [WEIGHT_WIDTH-1:0]      debug_weight_min,
    output reg  [WEIGHT_WIDTH-1:0]      debug_weight_max,
    output reg  [31:0]                  debug_weight_sum
);

    localparam [WEIGHT_WIDTH-1:0] W_MIN_VALUE = W_MIN;
    localparam [WEIGHT_WIDTH-1:0] W_MAX_VALUE = W_MAX;

    reg [TIME_WIDTH-1:0] current_time;
    always @(posedge clk) begin
        if (!rst_n)
            current_time <= {TIME_WIDTH{1'b0}};
        else if (enable)
            current_time <= current_time + {{(TIME_WIDTH-1){1'b0}}, 1'b1};
    end

    //-------------------------------------------------------------------------
    // Existing core_group used as an input spike front-end.
    //-------------------------------------------------------------------------
    wire cg_ext_ready;
    wire cg_out_valid;
    wire [CORE_ID_WIDTH-1:0] cg_out_id;
    wire [15:0] cg_spike_count;
    wire cg_busy;

    core_group #(
        .GROUP_ID(0),
        .NEURONS_PER_GROUP(CORE_NEURONS),
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(8),
        .THRESHOLD_WIDTH(16),
        .LEAK_WIDTH(8),
        .REFRAC_WIDTH(8),
        .SPIKE_BUFFER_DEPTH(64)
    ) u_core_group_frontend (
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
        .global_threshold(16'd8),
        .global_leak_rate(8'd0),
        .global_refrac_period(8'd2),
        .weight_we(1'b0),
        .weight_src_id({CORE_ID_WIDTH{1'b0}}),
        .weight_dst_id({CORE_ID_WIDTH{1'b0}}),
        .weight_data(8'd0),
        .weight_exc(1'b1),
        .spike_count(cg_spike_count),
        .group_busy(cg_busy)
    );

    //-------------------------------------------------------------------------
    // Dense 64x10 classifier state.
    //-------------------------------------------------------------------------
    reg [WEIGHT_WIDTH-1:0] weights [0:OUTPUT_NEURONS-1][0:INPUT_NEURONS-1];
    reg [MEMBRANE_WIDTH-1:0] membrane [0:OUTPUT_NEURONS-1];
    reg [EVENT_WEIGHT_WIDTH:0] input_trace [0:INPUT_NEURONS-1];
    reg [15:0] homeostasis [0:OUTPUT_NEURONS-1];

    reg [TIME_WIDTH-1:0] image_start_time;
    reg [OUTPUT_ID_WIDTH-1:0] pending_winner;
    reg [TIME_WIDTH-1:0] pending_latency;
    reg pending_decision;

    assign event_ready = enable && !pending_decision && !image_done && cg_ext_ready;

    reg wta_spike_valid;
    reg [OUTPUT_ID_WIDTH-1:0] wta_spike_id;
    wire wta_winner_valid;
    wire [OUTPUT_ID_WIDTH-1:0] wta_winner_id;
    wire wta_inhibit_active;

    winner_take_all #(
        .NUM_OUTPUTS(OUTPUT_NEURONS),
        .OUTPUT_ID_WIDTH(OUTPUT_ID_WIDTH),
        .INHIBIT_CYCLES(WTA_INHIBIT_CYCLES),
        .INHIBIT_CNT_WIDTH(8)
    ) u_wta (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .spike_valid(wta_spike_valid),
        .spike_neuron_id(wta_spike_id),
        .winner_valid(wta_winner_valid),
        .winner_neuron_id(wta_winner_id),
        .inhibit_active(wta_inhibit_active)
    );

    integer out_i;
    integer in_i;
    integer best_score;
    integer score;
    integer next_weight;
    integer updates_this_image;
    integer weight_sum_next;
    integer min_next;
    integer max_next;
    reg [OUTPUT_ID_WIDTH-1:0] best_out;

    always @(posedge clk) begin
        if (!rst_n) begin
            image_done           <= 1'b0;
            winner_valid         <= 1'b0;
            winner_id            <= {OUTPUT_ID_WIDTH{1'b0}};
            latency_cycles       <= {TIME_WIDTH{1'b0}};
            total_images         <= 32'd0;
            total_output_spikes  <= 32'd0;
            total_weight_updates <= 32'd0;
            debug_weight_min     <= W_MAX_VALUE;
            debug_weight_max     <= W_MIN_VALUE;
            debug_weight_sum     <= 32'd0;
            image_start_time     <= {TIME_WIDTH{1'b0}};
            pending_winner       <= {OUTPUT_ID_WIDTH{1'b0}};
            pending_latency      <= {TIME_WIDTH{1'b0}};
            pending_decision     <= 1'b0;
            wta_spike_valid      <= 1'b0;
            wta_spike_id         <= {OUTPUT_ID_WIDTH{1'b0}};

            for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                membrane[out_i]    <= {MEMBRANE_WIDTH{1'b0}};
                homeostasis[out_i] <= 16'd0;
                for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1) begin
                    weights[out_i][in_i] <= 4 + ((out_i * 17 + in_i * 5 + (in_i >> 3) * 3) % 8);
                end
            end
            for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                input_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
        end else begin
            image_done      <= 1'b0;
            winner_valid    <= 1'b0;
            wta_spike_valid <= 1'b0;

            if (image_start) begin
                image_start_time <= current_time;
                pending_decision <= 1'b0;
                for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1)
                    membrane[out_i] <= {MEMBRANE_WIDTH{1'b0}};
                for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1)
                    input_trace[in_i] <= {EVENT_WEIGHT_WIDTH+1{1'b0}};
            end

            if (event_valid && event_ready) begin
                if (input_trace[event_neuron_id] + event_weight > 15)
                    input_trace[event_neuron_id] <= 5'd15;
                else
                    input_trace[event_neuron_id] <= input_trace[event_neuron_id] + event_weight;

                for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                    membrane[out_i] <= membrane[out_i] +
                        (weights[out_i][event_neuron_id] * event_weight);
                end
            end

            if (image_end && !pending_decision) begin
                best_out = {OUTPUT_ID_WIDTH{1'b0}};
                best_score = -2147483647;
                for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                    score = membrane[out_i] - (homeostasis[out_i] << HOMEOSTASIS_SHIFT);
                    if (score > best_score) begin
                        best_score = score;
                        best_out = out_i;
                    end
                end

                pending_winner   <= best_out;
                pending_latency  <= current_time - image_start_time;
                pending_decision <= 1'b1;
                wta_spike_valid  <= 1'b1;
                wta_spike_id     <= best_out;
            end

            if (pending_decision && wta_winner_valid) begin
                pending_decision    <= 1'b0;
                image_done          <= 1'b1;
                winner_valid        <= 1'b1;
                winner_id           <= wta_winner_id;
                latency_cycles      <= pending_latency;
                total_images        <= total_images + 32'd1;
                total_output_spikes <= total_output_spikes + 32'd1;

                if (learning_enable) begin
                    updates_this_image = 0;

                    for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                        if (homeostasis[out_i] > HOMEOSTASIS_DECAY)
                            homeostasis[out_i] <= homeostasis[out_i] - HOMEOSTASIS_DECAY;
                        else
                            homeostasis[out_i] <= 16'd0;
                    end
                    homeostasis[wta_winner_id] <= homeostasis[wta_winner_id] + HOMEOSTASIS_INC;

                    for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1) begin
                        if (input_trace[in_i] != 0) begin
                            next_weight = weights[wta_winner_id][in_i] + A_PLUS + (input_trace[in_i] >> 3);
                            if (next_weight > W_MAX)
                                next_weight = W_MAX;
                        end else begin
                            next_weight = weights[wta_winner_id][in_i] - A_MINUS;
                            if (next_weight < W_MIN)
                                next_weight = W_MIN;
                        end

                        if (next_weight != weights[wta_winner_id][in_i])
                            updates_this_image = updates_this_image + 1;
                        weights[wta_winner_id][in_i] <= next_weight[WEIGHT_WIDTH-1:0];
                    end
                    total_weight_updates <= total_weight_updates + updates_this_image;
                end

                weight_sum_next = 0;
                min_next = W_MAX;
                max_next = W_MIN;
                for (out_i = 0; out_i < OUTPUT_NEURONS; out_i = out_i + 1) begin
                    for (in_i = 0; in_i < INPUT_NEURONS; in_i = in_i + 1) begin
                        weight_sum_next = weight_sum_next + weights[out_i][in_i];
                        if (weights[out_i][in_i] < min_next)
                            min_next = weights[out_i][in_i];
                        if (weights[out_i][in_i] > max_next)
                            max_next = weights[out_i][in_i];
                    end
                end
                debug_weight_sum <= weight_sum_next[31:0];
                debug_weight_min <= min_next[WEIGHT_WIDTH-1:0];
                debug_weight_max <= max_next[WEIGHT_WIDTH-1:0];
            end
        end
    end

endmodule
