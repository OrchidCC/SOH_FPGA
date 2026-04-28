// =============================================================================
// mac3_pack_tb.v — Testbench for 3-packed MAC unit
// =============================================================================

`timescale 1ns / 1ps

module mac3_pack_tb;

    reg clk;
    reg en;
    reg signed [3:0] w1, w2, w3, x;
    wire signed [7:0] p1, p2, p3;

    mac3_pack dut (
        .clk(clk), .en(en),
        .w1(w1), .w2(w2), .w3(w3), .x(x),
        .p1(p1), .p2(p2), .p3(p3)
    );

    // Clock
    always #5 clk = ~clk;

    // Expected results
    integer exp1, exp2, exp3;
    integer errors;

    task check;
        input signed [3:0] tw1, tw2, tw3, tx;
        begin
            w1 = tw1; w2 = tw2; w3 = tw3; x = tx;
            en = 1;
            @(posedge clk); #1;
            @(posedge clk); #1;  // 2-cycle MAC pipeline
            en = 0;
            @(posedge clk); #1;  // wait for output register

            exp1 = tw1 * tx;
            exp2 = tw2 * tx;
            exp3 = tw3 * tx;

            if (p1 !== exp1[7:0] || p2 !== exp2[7:0] || p3 !== exp3[7:0]) begin
                $display("FAIL: w=(%0d,%0d,%0d) x=%0d -> p=(%0d,%0d,%0d) exp=(%0d,%0d,%0d)",
                         tw1, tw2, tw3, tx, p1, p2, p3, exp1, exp2, exp3);
                errors = errors + 1;
            end
        end
    endtask

    integer i, j;
    initial begin
        clk = 0; en = 0; errors = 0;
        w1 = 0; w2 = 0; w3 = 0; x = 0;
        #20;

        // --- Basic cases ---
        check(4'sd1,  4'sd2,  4'sd3,  4'sd4);   // all positive
        check(-4'sd1, -4'sd2, -4'sd3, 4'sd4);   // negative weights
        check(4'sd7,  4'sd7,  4'sd7,  4'sd7);   // max positive
        check(-4'sd7, -4'sd7, -4'sd7, -4'sd7);  // max negative * max negative
        check(4'sd7,  -4'sd7, 4'sd7,  -4'sd7);  // mixed signs
        check(4'sd0,  4'sd0,  4'sd0,  4'sd5);   // zero weights
        check(4'sd3,  4'sd3,  4'sd3,  4'sd0);   // zero activation

        // --- Exhaustive: sweep all w1 and x values ---
        for (i = -7; i <= 7; i = i + 1) begin
            for (j = -7; j <= 7; j = j + 1) begin
                check(i[3:0], 4'sd3, -4'sd5, j[3:0]);
            end
        end

        $display("\n=== MAC3 Pack Test Complete ===");
        if (errors == 0)
            $display("ALL TESTS PASSED");
        else
            $display("%0d ERRORS", errors);

        $finish;
    end

endmodule
