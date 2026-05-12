//=============================================================================
// Simple winner-take-all helper
//
// This module is intentionally small and beginner-readable. When enabled, the
// first output spike observed while no inhibition window is active becomes the
// winner. Other output neurons are suppressed until the window expires.
//
// The first unsupervised MNIST smoke test has one output neuron, so it does not
// need WTA yet. This helper is provided for the next multi-output stage.
//=============================================================================

`timescale 1ns / 1ps

module winner_take_all #(
    parameter NUM_OUTPUTS       = 4,
    parameter OUTPUT_ID_WIDTH   = $clog2(NUM_OUTPUTS),
    parameter INHIBIT_CYCLES    = 16,
    parameter INHIBIT_CNT_WIDTH = 8
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       enable,
    input  wire                       spike_valid,
    input  wire [OUTPUT_ID_WIDTH-1:0] spike_neuron_id,
    output reg                        winner_valid,
    output reg  [OUTPUT_ID_WIDTH-1:0] winner_neuron_id,
    output reg                        inhibit_active
);

    localparam [INHIBIT_CNT_WIDTH-1:0] INHIBIT_RELOAD = INHIBIT_CYCLES;

    reg [INHIBIT_CNT_WIDTH-1:0] inhibit_count;

    always @(posedge clk) begin
        if (!rst_n) begin
            winner_valid     <= 1'b0;
            winner_neuron_id <= {OUTPUT_ID_WIDTH{1'b0}};
            inhibit_active   <= 1'b0;
            inhibit_count    <= {INHIBIT_CNT_WIDTH{1'b0}};
        end else begin
            winner_valid <= 1'b0;

            if (!enable) begin
                inhibit_active <= 1'b0;
                inhibit_count  <= {INHIBIT_CNT_WIDTH{1'b0}};
            end else if (inhibit_active) begin
                if (inhibit_count == 0) begin
                    inhibit_active <= 1'b0;
                end else begin
                    inhibit_count <= inhibit_count - {{(INHIBIT_CNT_WIDTH-1){1'b0}}, 1'b1};
                end
            end else if (spike_valid) begin
                winner_valid     <= 1'b1;
                winner_neuron_id <= spike_neuron_id;
                inhibit_active   <= 1'b1;
                inhibit_count    <= INHIBIT_RELOAD;
            end
        end
    end

endmodule
