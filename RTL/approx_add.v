module approx_add (
    input  [7:0] A,
    input  [7:0] B,
    output [7:0] SUM
);
    wire [7:0] exact_sum;
    assign exact_sum = A + B;

    // Approximate: truncate carry from bits 2 to 5
    assign SUM[1:0] = exact_sum[1:0];
    assign SUM[7:2] = A[7:2] | B[7:2]; // Use OR as a fast approximation

endmodule
