// =============================================================================
// layer_stage.v — Output-Neuron-Parallel FC Layer with Correction-Free MAC
// =============================================================================
//
// KEY ARCHITECTURE:
//   Each DSP processes 3 output neurons in parallel (shared input x[col]).
//   N_PE DSPs → 3*N_PE neurons per batch.
//   col iterates 0..IN_DIM-1 (step 1, no tail handling needed).
//
// CORRECTION-FREE DATAPATH:
//   MAC outputs raw unsigned products (w_iu * x_u, no offset correction).
//   Correction deferred to post-accumulation:
//     result[j] = raw_acc[j] + modified_bias[j] - 8 * x_sum
//   where:
//     modified_bias[j] = original_bias[j] - 8*sum(w_iu[j]) + 64*IN_DIM
//     x_sum = sum(x_iu) over all inputs (computed once, shared)
//
// PER-CHANNEL PROGRAMMABLE ACTIVATION LUT:
//   Each neuron slot has its own 16-entry LUT (fused sin+GridWarp).
//   3 LUTs per DSP, loaded from PS.
//
// Memory map (wld_addr):
//   [0, W_TOTAL)                          → weights (3 RAMs per DSP)
//   [W_TOTAL, W_TOTAL+B_TOTAL)            → modified biases
//   [W_TOTAL+B_TOTAL, ..+OUT_DIM*16)      → act LUT (if HAS_SIN)
//
// =============================================================================

module layer_stage #(
    parameter IN_DIM  = 60,
    parameter OUT_DIM = 60,
    parameter N_PE    = 4,
    parameter HAS_SIN = 1,
    parameter IS_LAST = 0,
    parameter RSHIFT  = 4,
    parameter ACC_W   = 20,
    parameter W_TOTAL = 3600,
    parameter B_TOTAL = 60
)(
    input  wire clk,
    input  wire rst_n,

    input  wire [IN_DIM*4-1:0] in_vec,
    input  wire        in_valid,
    output reg         in_ack,

    output reg [OUT_DIM*4-1:0] out_vec,
    output reg         out_valid,
    input  wire        out_ack,

    output reg signed [ACC_W-1:0] raw_out,
    output reg         raw_valid,

    input  wire        wld_en,
    input  wire [12:0] wld_addr,
    input  wire [15:0] wld_data
);

    // ─── Derived parameters ──────────────────────────────────────
    localparam NEURONS_PER_DSP = 3;
    localparam NEURONS_PER_BATCH = N_PE * NEURONS_PER_DSP;
    localparam N_BATCHES = (OUT_DIM + NEURONS_PER_BATCH - 1) / NEURONS_PER_BATCH;
    localparam ROWS_PER_PE = N_BATCHES * NEURONS_PER_DSP;  // max neurons assigned to one DSP
    localparam W_PER_SLOT  = IN_DIM;  // weights per neuron slot

    // Activation LUT
    localparam ACT_LUT_BASE  = W_TOTAL + B_TOTAL;
    localparam ACT_LUT_TOTAL = OUT_DIM * 16;
    localparam ACT_PER_PE    = ROWS_PER_PE * 16;

    // ─── Bias storage (modified bias: bias - 8*Σwiu + 64*IN_DIM) ─
    reg signed [ACC_W-1:0] b_mem [0:B_TOTAL-1];

    always @(posedge clk) begin
        if (wld_en && wld_addr >= W_TOTAL && wld_addr < W_TOTAL + B_TOTAL)
            b_mem[wld_addr - W_TOTAL] <= {{(ACC_W-16){wld_data[15]}}, wld_data};
    end

    // ─── Weight load distribution ────────────────────────────────
    // Weights loaded in order: for each batch, for each DSP, for each slot (0,1,2),
    // IN_DIM weights sequentially.
    // Total: N_BATCHES * N_PE * 3 * IN_DIM = W_TOTAL
    reg [5:0]  ld_col;       // 0..IN_DIM-1
    reg [1:0]  ld_slot;      // 0..2 (which of 3 neuron slots)
    reg [4:0]  ld_pe;        // 0..N_PE-1
    reg [3:0]  ld_batch;     // 0..N_BATCHES-1

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ld_col <= 0; ld_slot <= 0; ld_pe <= 0; ld_batch <= 0;
        end else if (wld_en && wld_addr < W_TOTAL) begin
            if (ld_col == IN_DIM - 1) begin
                ld_col <= 0;
                if (ld_slot == 2) begin
                    ld_slot <= 0;
                    if (ld_pe == N_PE - 1) begin
                        ld_pe <= 0;
                        ld_batch <= ld_batch + 1;
                    end else
                        ld_pe <= ld_pe + 1;
                end else
                    ld_slot <= ld_slot + 1;
            end else
                ld_col <= ld_col + 1;
        end
    end

    // ─── Act LUT load counters ───────────────────────────────────
    reg [3:0] ld_act_v;      // 0..15
    reg [1:0] ld_act_slot;   // 0..2
    reg [4:0] ld_act_pe;     // 0..N_PE-1
    reg [3:0] ld_act_batch;  // which batch's neurons

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ld_act_v <= 0; ld_act_slot <= 0; ld_act_pe <= 0; ld_act_batch <= 0;
        end else if (wld_en && wld_addr >= ACT_LUT_BASE
                            && wld_addr <  ACT_LUT_BASE + ACT_LUT_TOTAL) begin
            if (ld_act_v == 4'd15) begin
                ld_act_v <= 0;
                if (ld_act_slot == 2) begin
                    ld_act_slot <= 0;
                    if (ld_act_pe == N_PE - 1) begin
                        ld_act_pe <= 0;
                        ld_act_batch <= ld_act_batch + 1;
                    end else
                        ld_act_pe <= ld_act_pe + 1;
                end else
                    ld_act_slot <= ld_act_slot + 1;
            end else
                ld_act_v <= ld_act_v + 1;
        end
    end

    // ─── Input latch ─────────────────────────────────────────────
    reg [IN_DIM*4-1:0] x_packed;
    reg x_loaded;

    function signed [3:0] get_elem;
        input [IN_DIM*4-1:0] vec;
        input integer idx;
        begin
            get_elem = vec[idx*4 +: 4];
        end
    endfunction

    // ─── PE array signals ────────────────────────────────────────
    // Each DSP: 3 weights (one per neuron slot) + 1 shared input
    reg  signed [3:0] pe_w1 [0:N_PE-1];  // slot 0 weight
    reg  signed [3:0] pe_w2 [0:N_PE-1];  // slot 1 weight
    reg  signed [3:0] pe_w3 [0:N_PE-1];  // slot 2 weight
    reg  signed [3:0] pe_x;              // shared input
    reg               pe_en;

    // Raw unsigned outputs from mac3_pack
    wire [7:0] pe_p1 [0:N_PE-1];
    wire [7:0] pe_p2 [0:N_PE-1];
    wire [7:0] pe_p3 [0:N_PE-1];

    // Per-PE weight read wires (exposed from generate block)
    wire signed [3:0] pe_w_rd_s0 [0:N_PE-1];
    wire signed [3:0] pe_w_rd_s1 [0:N_PE-1];
    wire signed [3:0] pe_w_rd_s2 [0:N_PE-1];

    // 3 accumulators per DSP (unsigned, for raw products)
    reg [ACC_W-1:0] acc_s0 [0:N_PE-1];
    reg [ACC_W-1:0] acc_s1 [0:N_PE-1];
    reg [ACC_W-1:0] acc_s2 [0:N_PE-1];

    // Shared x_sum accumulator (Σ x_iu over all inputs)
    reg [ACC_W-1:0] x_sum;

    // ─── FSM ─────────────────────────────────────────────────────
    localparam S_IDLE   = 0,
               S_STREAM = 1,
               S_DRAIN  = 2,
               S_POST   = 3,
               S_HOLD   = 4;
    reg [2:0] state;

    reg [5:0] batch_row;   // first neuron index of current batch
    reg [3:0] batch_cnt;   // which batch (for weight/LUT addressing)
    reg [5:0] col;         // current input column (0..IN_DIM-1)
    reg       pipe_d1;
    reg       pipe_valid;
    reg       post_wait;   // extra cycle for requant pipeline

    // Per-PE post-processing signals
    wire signed [3:0] pe_post_s0 [0:N_PE-1];
    wire signed [3:0] pe_post_s1 [0:N_PE-1];
    wire signed [3:0] pe_post_s2 [0:N_PE-1];

    // ─── Per-PE: 3 weight RAMs + 3 act LUTs + MAC ───────────────
    genvar gi;
    generate
        for (gi = 0; gi < N_PE; gi = gi + 1) begin : gen_pe

            // ── 3 weight RAMs (one per neuron slot) ──
            // Each: IN_DIM deep × 4-bit, addressed by col
            (* ram_style = "distributed" *)
            reg signed [3:0] w_mem_s0 [0:N_BATCHES*W_PER_SLOT-1];
            (* ram_style = "distributed" *)
            reg signed [3:0] w_mem_s1 [0:N_BATCHES*W_PER_SLOT-1];
            (* ram_style = "distributed" *)
            reg signed [3:0] w_mem_s2 [0:N_BATCHES*W_PER_SLOT-1];

            // Weight load: route to correct slot RAM
            wire [12:0] wld_local = ld_batch * W_PER_SLOT + ld_col;
            always @(posedge clk) begin
                if (wld_en && wld_addr < W_TOTAL && ld_pe == gi) begin
                    case (ld_slot)
                        2'd0: w_mem_s0[wld_local] <= wld_data[3:0];
                        2'd1: w_mem_s1[wld_local] <= wld_data[3:0];
                        2'd2: w_mem_s2[wld_local] <= wld_data[3:0];
                    endcase
                end
            end

            // Weight read (addressed by batch_cnt * IN_DIM + col)
            wire [12:0] w_rd_addr = batch_cnt * W_PER_SLOT + col;
            assign pe_w_rd_s0[gi] = w_mem_s0[w_rd_addr];
            assign pe_w_rd_s1[gi] = w_mem_s1[w_rd_addr];
            assign pe_w_rd_s2[gi] = w_mem_s2[w_rd_addr];

            // ── Per-slot requant — Stage 1: bias read + add-sub (registered) ──
            reg signed [ACC_W-1:0] corr_r_s0, corr_r_s1, corr_r_s2;
            always @(posedge clk) begin
                corr_r_s0 <= $signed(acc_s0[gi]) + b_mem[batch_row + gi*3]     - $signed(x_sum << 3);
                corr_r_s1 <= $signed(acc_s1[gi]) + b_mem[batch_row + gi*3 + 1] - $signed(x_sum << 3);
                corr_r_s2 <= $signed(acc_s2[gi]) + b_mem[batch_row + gi*3 + 2] - $signed(x_sum << 3);
            end

            // ── Per-slot requant — Stage 2: shift + clamp (registered) ──
            wire signed [ACC_W-1:0] rq_s0 = corr_r_s0 >>> RSHIFT;
            wire signed [ACC_W-1:0] rq_s1 = corr_r_s1 >>> RSHIFT;
            wire signed [ACC_W-1:0] rq_s2 = corr_r_s2 >>> RSHIFT;

            wire signed [3:0] clamp_s0 = (rq_s0 > 7) ? 4'sd7 : (rq_s0 < -7) ? -4'sd7 : rq_s0[3:0];
            wire signed [3:0] clamp_s1 = (rq_s1 > 7) ? 4'sd7 : (rq_s1 < -7) ? -4'sd7 : rq_s1[3:0];
            wire signed [3:0] clamp_s2 = (rq_s2 > 7) ? 4'sd7 : (rq_s2 < -7) ? -4'sd7 : rq_s2[3:0];

            reg signed [3:0] rq_r_s0, rq_r_s1, rq_r_s2;
            always @(posedge clk) begin
                rq_r_s0 <= clamp_s0;
                rq_r_s1 <= clamp_s1;
                rq_r_s2 <= clamp_s2;
            end

            // ── 3 activation LUTs (one per neuron slot) ──
            if (HAS_SIN) begin : has_lut
                (* ram_style = "distributed" *)
                reg signed [3:0] lut_s0 [0:N_BATCHES*16-1];
                (* ram_style = "distributed" *)
                reg signed [3:0] lut_s1 [0:N_BATCHES*16-1];
                (* ram_style = "distributed" *)
                reg signed [3:0] lut_s2 [0:N_BATCHES*16-1];

                // LUT load
                wire [9:0] lut_waddr = ld_act_batch * 16 + ld_act_v;
                always @(posedge clk) begin
                    if (wld_en && wld_addr >= ACT_LUT_BASE
                              && wld_addr <  ACT_LUT_BASE + ACT_LUT_TOTAL
                              && ld_act_pe == gi) begin
                        case (ld_act_slot)
                            2'd0: lut_s0[lut_waddr] <= wld_data[3:0];
                            2'd1: lut_s1[lut_waddr] <= wld_data[3:0];
                            2'd2: lut_s2[lut_waddr] <= wld_data[3:0];
                        endcase
                    end
                end

                // LUT read: signed → unsigned index
                wire [3:0] idx_s0 = {~rq_r_s0[3], rq_r_s0[2:0]};
                wire [3:0] idx_s1 = {~rq_r_s1[3], rq_r_s1[2:0]};
                wire [3:0] idx_s2 = {~rq_r_s2[3], rq_r_s2[2:0]};

                assign pe_post_s0[gi] = lut_s0[batch_cnt * 16 + idx_s0];
                assign pe_post_s1[gi] = lut_s1[batch_cnt * 16 + idx_s1];
                assign pe_post_s2[gi] = lut_s2[batch_cnt * 16 + idx_s2];
            end else begin : no_lut
                assign pe_post_s0[gi] = rq_r_s0;
                assign pe_post_s1[gi] = rq_r_s1;
                assign pe_post_s2[gi] = rq_r_s2;
            end

            // ── MAC unit ──
            mac3_pack u_mac (
                .clk(clk), .en(pe_en),
                .w1(pe_w1[gi]), .w2(pe_w2[gi]), .w3(pe_w3[gi]),
                .x(pe_x),
                .p1(pe_p1[gi]), .p2(pe_p2[gi]), .p3(pe_p3[gi])
            );
        end
    endgenerate

    // ─── FSM ─────────────────────────────────────────────────────
    reg signed [ACC_W-1:0] acc_biased;
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            in_ack <= 0; out_valid <= 0; raw_valid <= 0;
            pe_en <= 0; pipe_valid <= 0; pipe_d1 <= 0;
            x_loaded <= 0; batch_row <= 0; col <= 0;
            batch_cnt <= 0; x_sum <= 0; post_wait <= 0;
        end else begin
            in_ack <= 0;
            raw_valid <= 0;
            pipe_d1 <= pe_en;
            pipe_valid <= pipe_d1;

            case (state)
                S_IDLE: begin
                    pe_en <= 0;
                    if (in_valid && !x_loaded) begin
                        x_packed <= in_vec;
                        x_loaded <= 1;
                        in_ack <= 1;
                        batch_row <= 0;
                        batch_cnt <= 0;
                        col <= 0;
                        x_sum <= 0;
                        for (k = 0; k < N_PE; k = k + 1) begin
                            acc_s0[k] <= 0;
                            acc_s1[k] <= 0;
                            acc_s2[k] <= 0;
                        end
                        state <= S_STREAM;
                    end
                    if (out_ack && out_valid)
                        out_valid <= 0;
                end

                S_STREAM: begin
                    // Accumulate raw unsigned products (2-cycle MAC latency)
                    if (pipe_valid) begin
                        for (k = 0; k < N_PE; k = k + 1) begin
                            acc_s0[k] <= acc_s0[k] + {12'b0, pe_p1[k]};
                            acc_s1[k] <= acc_s1[k] + {12'b0, pe_p2[k]};
                            acc_s2[k] <= acc_s2[k] + {12'b0, pe_p3[k]};
                        end
                    end

                    if (col < IN_DIM) begin
                        pe_en <= 1;
                        pe_x <= get_elem(x_packed, col);

                        // Accumulate x_sum: x_iu = x_signed + 8
                        if (batch_cnt == 0)  // only compute x_sum in first batch
                            x_sum <= x_sum + {{(ACC_W-4){1'b0}}, get_elem(x_packed, col) + 4'd8};

                        // Load weights: 3 slots per DSP, same col
                        for (k = 0; k < N_PE; k = k + 1) begin
                            if (batch_row + k*3 < OUT_DIM) begin
                                pe_w1[k] <= pe_w_rd_s0[k];
                                pe_w2[k] <= (batch_row + k*3 + 1 < OUT_DIM) ? pe_w_rd_s1[k] : 4'sd0;
                                pe_w3[k] <= (batch_row + k*3 + 2 < OUT_DIM) ? pe_w_rd_s2[k] : 4'sd0;
                            end else begin
                                pe_w1[k] <= 0; pe_w2[k] <= 0; pe_w3[k] <= 0;
                            end
                        end
                        col <= col + 1;
                    end else begin
                        pe_en <= 0;
                        state <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    if (pipe_valid) begin
                        for (k = 0; k < N_PE; k = k + 1) begin
                            acc_s0[k] <= acc_s0[k] + {12'b0, pe_p1[k]};
                            acc_s1[k] <= acc_s1[k] + {12'b0, pe_p2[k]};
                            acc_s2[k] <= acc_s2[k] + {12'b0, pe_p3[k]};
                        end
                    end
                    if (!pipe_d1 && !pipe_valid)
                        state <= S_POST;
                end

                S_POST: begin
                    if (IS_LAST) begin
                        // Final layer: raw corrected output
                        for (k = 0; k < N_PE; k = k + 1) begin
                            if (batch_row + k*3 < OUT_DIM) begin
                                acc_biased = $signed(acc_s0[k]) + b_mem[batch_row + k*3] - $signed(x_sum << 3);
                                raw_out <= acc_biased;
                                raw_valid <= 1;
                            end
                        end
                        x_loaded <= 0;
                        state <= S_IDLE;
                    end else begin
                        if (!post_wait) begin
                            // Cycle 1: corr_r_sX registers (pipeline stage 1)
                            // Cycle 2: rq_r_sX registers (pipeline stage 2)
                            post_wait <= 1;
                        end else begin
                            post_wait <= 0;
                            // pe_post_sX now valid (from rq_r_sX via LUT)
                            for (k = 0; k < N_PE; k = k + 1) begin
                                if (batch_row + k*3 < OUT_DIM)
                                    out_vec[(batch_row+k*3)*4 +: 4] <= pe_post_s0[k];
                                if (batch_row + k*3 + 1 < OUT_DIM)
                                    out_vec[(batch_row+k*3+1)*4 +: 4] <= pe_post_s1[k];
                                if (batch_row + k*3 + 2 < OUT_DIM)
                                    out_vec[(batch_row+k*3+2)*4 +: 4] <= pe_post_s2[k];
                            end

                            if (batch_row + NEURONS_PER_BATCH >= OUT_DIM) begin
                                out_valid <= 1;
                                state <= S_HOLD;
                            end else begin
                                batch_row <= batch_row + NEURONS_PER_BATCH;
                                batch_cnt <= batch_cnt + 1;
                                col <= 0;
                                for (k = 0; k < N_PE; k = k + 1) begin
                                    acc_s0[k] <= 0;
                                    acc_s1[k] <= 0;
                                    acc_s2[k] <= 0;
                                end
                                state <= S_STREAM;
                            end
                        end
                    end
                end

                S_HOLD: begin
                    if (out_ack) begin
                        out_valid <= 0;
                        x_loaded <= 0;
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
