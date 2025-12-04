// Synth-safe, balanced adder tree popcount.
// Vivado infers LUT adders. W must be >=1.
module xnor_popcount #(
    parameter integer W = 256
)(
   input  wire [W-1:0] a,
   input  wire [W-1:0] b,
   output wire [$clog2(W+1)-1:0] sum
);

//wire [W-1:0] a;
//wire [W-1:0] b;
    wire [W-1:0] x = ~(a ^ b);
    

    // Simple iterative adder tree
    function integer CLOG2; input integer v; integer i; begin
        i=0; v=v-1; for(i=0; v>0; i=i+1) v=v>>1; CLOG2=i;
    end endfunction

    // Build reduction levels
    localparam integer STAGES = CLOG2(W);
    wire [W-1:0] stage [0:STAGES];

    assign stage[0] = x;
    genvar s,i;
    generate
        for (s=0; s<STAGES; s=s+1) begin : STG
            localparam integer INW  = (W>>s) + ((W & ((1<<s)-1)) ? 1 : 0);
            localparam integer OUTW = (INW+1)>>1;
            for (i=0; i<OUTW; i=i+1) begin : ADD
                wire [1:0] pair = { (2*i+1<INW)? stage[s][2*i+1] : 1'b0, stage[s][2*i] };
                assign stage[s+1][i] = pair[0] + pair[1];
            end
        end
    endgenerate

    assign sum = stage[STAGES][$clog2(W+1)-1:0];
endmodule
