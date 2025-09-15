module cim_memory #(parameter N = 8)(
    input wire [N-1:0] input_vector,
    input wire [N-1:0] weight_vector,
    output wire signed [7:0] dot_val
);
    integer i;
    reg [N-1:0] xnor_result;
    reg [$clog2(N+1)-1:0] popcount_out;
    

    always @(*) begin
        xnor_result = ~(input_vector ^ weight_vector);
        popcount_out = 0;
//        for (i = 0; i < N; i = i + 1) begin
//            popcount_out = popcount_out + xnor_result[i];
//        end
        while(xnor_result)
        begin
           xnor_result = xnor_result & (xnor_result - 1);
           popcount_out = popcount_out + 1;
        end
     end
       
assign dot_val = (popcount_out << 1) - N;   // dot = 2*popcount - N

endmodule
