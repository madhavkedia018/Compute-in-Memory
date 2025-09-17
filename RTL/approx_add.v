`timescale 1ns/1ps

module approx_add (
    input  [7:0] A,
    input  [7:0] B,
    output [8:0] SUM
);
    // Approximate addition for lower 6 bits
    assign SUM[5:0] = A[5:0] | B[5:0];   // OR-based approximation

    // Exact addition for upper 2 bits (no carry from lower part)
    assign SUM[8:6] = A[7:6] + B[7:6];   // carry from lower 2 ignored
endmodule

