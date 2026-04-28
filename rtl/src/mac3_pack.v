// =============================================================================
// mac3_pack.v — 3 W4A4 MACs packed in 1 DSP48E1 (correction-free)
// =============================================================================
//
// Packs 3 unsigned 4-bit multiplications into one DSP48E1.
// Offset encoding: w_signed + 8 → w_unsigned, x_signed + 8 → x_unsigned.
// Outputs RAW unsigned sub-products (no offset correction).
// Correction is deferred to post-accumulation in layer_stage.
//
// Usage: output-neuron-parallel packing for FC layers.
//   w1, w2, w3 = weights from 3 DIFFERENT output neurons at same input col
//   x          = shared input activation
//   p1, p2, p3 = raw unsigned products (w_iu × x_u), 8-bit each
//
// Latency: 2 cycles (multiply + output register)
// Resources: 1 DSP48E1, ~0 LUT (no correction logic)
// =============================================================================

module mac3_pack (
    input  wire        clk,
    input  wire        en,
    input  wire signed [3:0] w1, w2, w3,
    input  wire signed [3:0] x,
    output reg  [7:0] p1, p2, p3  // unsigned raw products
);

    // --- Stage 1: Offset encode + packed multiply (→ DSP48E1) ---
    wire [3:0] w1u = w1 + 4'd8;
    wire [3:0] w2u = w2 + 4'd8;
    wire [3:0] w3u = w3 + 4'd8;
    wire [3:0] xu  = x  + 4'd8;

    wire [24:0] a_packed = {3'b0, w3u, 5'b0, w2u, 5'b0, w1u};

    // This is the ONLY multiply → maps to 1 DSP48E1
    (* use_dsp = "yes" *)
    reg [28:0] product_reg;

    always @(posedge clk) begin
        if (en) begin
            product_reg <= a_packed * {25'b0, xu};
        end
    end

    // --- Stage 2: Extract raw unsigned sub-products (no correction) ---
    always @(posedge clk) begin
        if (en) begin
            p1 <= product_reg[7:0];
            p2 <= product_reg[16:9];
            p3 <= product_reg[25:18];
        end
    end

endmodule
