
module thresh_act #(
    parameter integer NCH  = 8,   // number of channels (PEs)
    parameter integer SUMW = 16   // accumulator width
)(
    input  wire [NCH*SUMW-1:0] sum_in,   // concatenated sums
    input  wire [NCH*SUMW-1:0] thresh,   // concatenated thresholds
    output wire [NCH-1:0]      act_bits  // 1-bit activations
);

genvar i;
generate
    for (i = 0; i < NCH; i = i + 1) begin : ACT
        wire [SUMW-1:0] sum_i    = sum_in[i*SUMW +: SUMW];
        wire [SUMW-1:0] thresh_i = thresh [i*SUMW +: SUMW];

        assign act_bits[i] = (sum_i >= thresh_i);
    end
endgenerate

endmodule
