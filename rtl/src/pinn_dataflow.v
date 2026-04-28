// =============================================================================
// pinn_dataflow.v — 5-Stage Dataflow PINN Inference Pipeline
// =============================================================================
//
//   features ──→ [L0] ──→ [L1] ──→ [L2] ──→ [L3] ──→ [L4] ──→ result
//                10PE     20PE     11PE     11PE      1PE
//               17→60    60→60    60→32    32→32    32→1
//               ~34cy    ~60cy    ~60cy    ~32cy    ~32cy
//               +sin     +sin              +sin
//
// Output-neuron-parallel DSP packing: each DSP handles 3 output neurons.
// Correction-free MAC: offset correction deferred to post-accumulation.
//
// Throughput: ~60 cycles/sample = 0.60 µs @ 100MHz
// DSP48E1:    53 total (24% of ZYNQ-7020's 220)
// =============================================================================

module pinn_dataflow (
    input  wire clk,
    input  wire rst_n,

    input  wire [67:0]  features_in,
    input  wire         in_valid,
    output wire         in_ready,

    output wire signed [19:0] result,
    output wire         result_valid,

    input  wire        wld_en,
    input  wire [2:0]  wld_layer,
    input  wire [12:0] wld_addr,
    input  wire [15:0] wld_data
);

    // ─── Inter-layer packed vectors ──────────────────────────────
    wire [239:0] l0_out;
    wire         l0_ov, l0_oa, l0_ia;

    wire [239:0] l1_out;
    wire         l1_ov, l1_oa, l1_ia;

    wire [127:0] l2_out;
    wire         l2_ov, l2_oa, l2_ia;

    wire [127:0] l3_out;
    wire         l3_ov, l3_oa, l3_ia;

    wire signed [19:0] l4_raw;
    wire               l4_rv, l4_ia;

    assign in_ready = ~u_L0.x_loaded;

    // ═══════════════════════════════════════════════════════════════
    // PE allocation (output-neuron-parallel: 3 neurons per DSP)
    //
    //   Layer  IN×OUT  PE  Neurons/batch  Batches  Cyc/batch  Total
    //   L0     17×60   10  30             2        17         ~34
    //   L1     60×60   20  60             1        60         ~60  ← bottleneck
    //   L2     60×32   11  33(→32)        1        60         ~60
    //   L3     32×32   11  33(→32)        1        32         ~32
    //   L4     32×1     1  3(→1)          1        32         ~32
    //
    //   Total DSP: 10+20+11+11+1 = 53 (24%)
    //   Throughput: ~60 cycles/sample = 0.60 µs @ 100MHz
    // ═══════════════════════════════════════════════════════════════

    // ─── L0: 17→60 + sin, 10 PEs ────────────────────────────────
    layer_stage #(
        .IN_DIM(17), .OUT_DIM(60), .N_PE(10),
        .HAS_SIN(1), .IS_LAST(0), .RSHIFT(4),
        .W_TOTAL(1020), .B_TOTAL(60)
    ) u_L0 (
        .clk(clk), .rst_n(rst_n),
        .in_vec(features_in),
        .in_valid(in_valid), .in_ack(l0_ia),
        .out_vec(l0_out), .out_valid(l0_ov), .out_ack(l0_oa),
        .raw_out(), .raw_valid(),
        .wld_en(wld_en && wld_layer==3'd0),
        .wld_addr(wld_addr), .wld_data(wld_data)
    );

    // ─── L1: 60→60 + sin, 20 PEs ────────────────────────────────
    layer_stage #(
        .IN_DIM(60), .OUT_DIM(60), .N_PE(20),
        .HAS_SIN(1), .IS_LAST(0), .RSHIFT(4),
        .W_TOTAL(3600), .B_TOTAL(60)
    ) u_L1 (
        .clk(clk), .rst_n(rst_n),
        .in_vec(l0_out), .in_valid(l0_ov), .in_ack(l0_oa),
        .out_vec(l1_out), .out_valid(l1_ov), .out_ack(l1_oa),
        .raw_out(), .raw_valid(),
        .wld_en(wld_en && wld_layer==3'd1),
        .wld_addr(wld_addr), .wld_data(wld_data)
    );

    // ─── L2: 60→32, no sin, 11 PEs ──────────────────────────────
    layer_stage #(
        .IN_DIM(60), .OUT_DIM(32), .N_PE(11),
        .HAS_SIN(0), .IS_LAST(0), .RSHIFT(4),
        .W_TOTAL(1920), .B_TOTAL(32)
    ) u_L2 (
        .clk(clk), .rst_n(rst_n),
        .in_vec(l1_out), .in_valid(l1_ov), .in_ack(l1_oa),
        .out_vec(l2_out), .out_valid(l2_ov), .out_ack(l2_oa),
        .raw_out(), .raw_valid(),
        .wld_en(wld_en && wld_layer==3'd2),
        .wld_addr(wld_addr), .wld_data(wld_data)
    );

    // ─── L3: 32→32 + sin, 11 PEs ────────────────────────────────
    layer_stage #(
        .IN_DIM(32), .OUT_DIM(32), .N_PE(11),
        .HAS_SIN(1), .IS_LAST(0), .RSHIFT(4),
        .W_TOTAL(1024), .B_TOTAL(32)
    ) u_L3 (
        .clk(clk), .rst_n(rst_n),
        .in_vec(l2_out), .in_valid(l2_ov), .in_ack(l2_oa),
        .out_vec(l3_out), .out_valid(l3_ov), .out_ack(l3_oa),
        .raw_out(), .raw_valid(),
        .wld_en(wld_en && wld_layer==3'd3),
        .wld_addr(wld_addr), .wld_data(wld_data)
    );

    // ─── L4: 32→1, no sin, 1 PE (output layer) ──────────────────
    layer_stage #(
        .IN_DIM(32), .OUT_DIM(1), .N_PE(1),
        .HAS_SIN(0), .IS_LAST(1), .RSHIFT(0),
        .W_TOTAL(32), .B_TOTAL(1)
    ) u_L4 (
        .clk(clk), .rst_n(rst_n),
        .in_vec(l3_out), .in_valid(l3_ov), .in_ack(l3_oa),
        .out_vec(), .out_valid(), .out_ack(1'b1),
        .raw_out(l4_raw), .raw_valid(l4_rv),
        .wld_en(wld_en && wld_layer==3'd4),
        .wld_addr(wld_addr), .wld_data(wld_data)
    );

    assign result = l4_raw;
    assign result_valid = l4_rv;

endmodule
