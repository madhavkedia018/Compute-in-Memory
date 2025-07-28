module approx_mult (
    input  [3:0] A,
    input  [3:0] B,
    output [7:0] P
);
    assign P = {A[3:2] * B[3:2], 4'b0000}; // Truncate lower bits for area savings
endmodule
