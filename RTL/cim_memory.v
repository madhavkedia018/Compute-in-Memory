module cim_memory #(parameter N = 8)(
    input wire [N-1:0] input_vector,
    input wire [N-1:0] weight_vector,
    output reg [$clog2(N+1)-1:0] popcount_out
);
    integer i;
    reg [N-1:0] xnor_result;

    always @(*) begin
        xnor_result = ~(input_vector ^ weight_vector);
        popcount_out = 0;
        for (i = 0; i < N; i = i + 1) begin
            popcount_out = popcount_out + xnor_result[i];
        end
    end
endmodule
