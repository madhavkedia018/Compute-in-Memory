`timescale 1ns/1ps

module tb_approx_add;

    reg  [7:0] A, B;
    wire [8:0] SUM;
    wire [8:0] exact_sum;   // 9 bits to avoid overflow

    integer i, j;
    integer total_tests;
    integer err_count;
    real    total_rel_error;  // accumulate relative error
    real    avg_rel_error;
    real rel_error;

    // DUT: Approximate Adder
    approx_add dut (
        .A(A),
        .B(B),
        .SUM(SUM)
    );

    // Exact reference
    assign exact_sum = A + B;

    initial begin
        total_tests     = 0;
        err_count       = 0;
        total_rel_error = 0.0;

        $display("===== Testing Proposed Approximate Adder (Relative Error) =====");

        // Exhaustive test: 65,536 cases (can switch to random if slow)
        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                A = i;
                B = j;
                #1; // settle

                total_tests = total_tests + 1;

                if (exact_sum != 0) begin  // avoid divide-by-zero
                    
                    rel_error = ( (exact_sum > SUM) ? 
                                   (exact_sum - SUM) : (SUM - exact_sum) )
                                   / (1.0 * exact_sum);
                    total_rel_error = total_rel_error + rel_error;

                    if (rel_error > 0)
                        err_count = err_count + 1;
                end
            end
        end

        // Average Relative Error
        avg_rel_error = (total_rel_error / total_tests) * 100.0; // percentage

        $display("Total Tests         = %0d", total_tests);
        $display("Error Count         = %0d", err_count);
        $display("Error Rate          = %.2f %%", (err_count*100.0)/total_tests);
        $display("Average Relative Err= %.4f %%", avg_rel_error);

        $display("===== Simulation End =====");
        $finish;
    end

endmodule
