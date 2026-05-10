//=============================================================================
// Simple pair-based STDP learning engine
//
// This module is intentionally small and direct. It observes one pre-synaptic
// spike stream and one post-synaptic spike stream, remembers the most recent
// spike time for each neuron, and emits at most one weight update for a recent
// pre/post pair.
//
// Rules:
//   LTP: pre before post within STDP_WINDOW -> weight increases by A_PLUS
//   LTD: post before pre within STDP_WINDOW -> weight decreases by A_MINUS
//
// The module does not store the whole weight matrix. The surrounding top gives
// it the current synapse weight for the tiny experiment being learned.
//=============================================================================

`timescale 1ns / 1ps

module stdp_learning_engine #(
    parameter NUM_NEURONS      = 16,
    parameter NEURON_ID_WIDTH  = 4,
    parameter WEIGHT_WIDTH     = 8,
    parameter TIME_WIDTH       = 16,
    parameter STDP_WINDOW      = 16,
    parameter A_PLUS           = 5,
    parameter A_MINUS          = 3,
    parameter W_MIN            = 0,
    parameter W_MAX            = 255
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    input  wire [TIME_WIDTH-1:0]        current_time,

    input  wire                         pre_spike_valid,
    input  wire [NEURON_ID_WIDTH-1:0]   pre_neuron_id,
    input  wire                         post_spike_valid,
    input  wire [NEURON_ID_WIDTH-1:0]   post_neuron_id,

    input  wire [WEIGHT_WIDTH-1:0]      current_weight,
    input  wire                         update_ready,

    output reg                          update_valid,
    output reg  [NEURON_ID_WIDTH-1:0]   update_src,
    output reg  [NEURON_ID_WIDTH-1:0]   update_dst,
    output reg  [WEIGHT_WIDTH-1:0]      update_weight,
    output reg                          update_exc,
    output reg signed [WEIGHT_WIDTH:0]  debug_delta,
    output reg  [1:0]                   debug_rule_applied
);

    localparam [1:0]
        RULE_NONE = 2'd0,
        RULE_LTP  = 2'd1,
        RULE_LTD  = 2'd2;

    reg [TIME_WIDTH-1:0] last_pre_time  [0:NUM_NEURONS-1];
    reg [TIME_WIDTH-1:0] last_post_time [0:NUM_NEURONS-1];
    reg                  last_pre_seen  [0:NUM_NEURONS-1];
    reg                  last_post_seen [0:NUM_NEURONS-1];

    reg [NEURON_ID_WIDTH-1:0] recent_pre_id;
    reg [NEURON_ID_WIDTH-1:0] recent_post_id;
    reg                       any_pre_seen;
    reg                       any_post_seen;

    integer init_i;
    integer dt;
    integer next_weight;

    // Emit a saturated LTP update for recent_pre_id -> post_id.
    task automatic emit_ltp;
        input [NEURON_ID_WIDTH-1:0] src_id;
        input [NEURON_ID_WIDTH-1:0] dst_id;
    begin
        next_weight = current_weight + A_PLUS;
        if (next_weight > W_MAX)
            next_weight = W_MAX;

        update_valid       <= 1'b1;
        update_src         <= src_id;
        update_dst         <= dst_id;
        update_weight      <= next_weight[WEIGHT_WIDTH-1:0];
        update_exc         <= 1'b1;
        debug_delta        <= $signed({1'b0, next_weight[WEIGHT_WIDTH-1:0]}) -
                              $signed({1'b0, current_weight});
        debug_rule_applied <= RULE_LTP;
    end
    endtask

    // Emit a saturated LTD update for src_id -> recent_post_id.
    task automatic emit_ltd;
        input [NEURON_ID_WIDTH-1:0] src_id;
        input [NEURON_ID_WIDTH-1:0] dst_id;
    begin
        next_weight = current_weight - A_MINUS;
        if (next_weight < W_MIN)
            next_weight = W_MIN;

        update_valid       <= 1'b1;
        update_src         <= src_id;
        update_dst         <= dst_id;
        update_weight      <= next_weight[WEIGHT_WIDTH-1:0];
        update_exc         <= 1'b1;
        debug_delta        <= $signed({1'b0, next_weight[WEIGHT_WIDTH-1:0]}) -
                              $signed({1'b0, current_weight});
        debug_rule_applied <= RULE_LTD;
    end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            update_valid       <= 1'b0;
            update_src         <= {NEURON_ID_WIDTH{1'b0}};
            update_dst         <= {NEURON_ID_WIDTH{1'b0}};
            update_weight      <= {WEIGHT_WIDTH{1'b0}};
            update_exc         <= 1'b1;
            debug_delta        <= {WEIGHT_WIDTH+1{1'b0}};
            debug_rule_applied <= RULE_NONE;
            recent_pre_id      <= {NEURON_ID_WIDTH{1'b0}};
            recent_post_id     <= {NEURON_ID_WIDTH{1'b0}};
            any_pre_seen       <= 1'b0;
            any_post_seen      <= 1'b0;

            for (init_i = 0; init_i < NUM_NEURONS; init_i = init_i + 1) begin
                last_pre_time[init_i]  <= {TIME_WIDTH{1'b0}};
                last_post_time[init_i] <= {TIME_WIDTH{1'b0}};
                last_pre_seen[init_i]  <= 1'b0;
                last_post_seen[init_i] <= 1'b0;
            end
        end else begin
            // Hold an update until the receiver accepts it.
            if (update_valid) begin
                if (update_ready)
                    update_valid <= 1'b0;
            end else begin
                debug_rule_applied <= RULE_NONE;
                debug_delta        <= {WEIGHT_WIDTH+1{1'b0}};

                if (enable) begin
                    // A pre spike can cause LTD if the post neuron fired first.
                    if (pre_spike_valid) begin
                        last_pre_time[pre_neuron_id] <= current_time;
                        last_pre_seen[pre_neuron_id] <= 1'b1;
                        recent_pre_id                <= pre_neuron_id;
                        any_pre_seen                 <= 1'b1;

                        if (any_post_seen &&
                            last_post_seen[recent_post_id] &&
                            current_time >= last_post_time[recent_post_id]) begin
                            dt = current_time - last_post_time[recent_post_id];
                            if (dt <= STDP_WINDOW)
                                emit_ltd(pre_neuron_id, recent_post_id);
                        end
                    end

                    // A post spike can cause LTP if the pre neuron fired first.
                    if (!update_valid && post_spike_valid) begin
                        last_post_time[post_neuron_id] <= current_time;
                        last_post_seen[post_neuron_id] <= 1'b1;
                        recent_post_id                 <= post_neuron_id;
                        any_post_seen                  <= 1'b1;

                        if (any_pre_seen &&
                            last_pre_seen[recent_pre_id] &&
                            current_time >= last_pre_time[recent_pre_id]) begin
                            dt = current_time - last_pre_time[recent_pre_id];
                            if (dt <= STDP_WINDOW)
                                emit_ltp(recent_pre_id, post_neuron_id);
                        end
                    end
                end
            end
        end
    end

endmodule
