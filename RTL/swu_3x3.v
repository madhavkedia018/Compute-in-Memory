// SWU: reads a stream of interleaved-channel pixels (one word per pixel)
// and emits successive 3x3 windows as concatenated words for MVTU.
// IMG_W: image width in pixels; CHW: channel pack width per pixel.
module swu_3x3 #(
    parameter integer IMG_W = 224,
    parameter integer CHW   = 64  // interleaved channels per pixel (bits)
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    input  wire [CHW-1:0]    pix_in,
    output reg                win_valid,
    output reg [9*CHW-1:0]   win_out // {w00,w01,w02,w10,...,w22}
);
    // 2 line buffers (BRAM)
    reg [CHW-1:0] lb0 [0:IMG_W-1];
    reg [CHW-1:0] lb1 [0:IMG_W-1];

    reg [$clog2(IMG_W):0] col;
    reg have_r1, have_r2;
    reg [CHW-1:0] w00,w01,w02,w10,w11,w12,w20,w21,w22;

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            col<=0; have_r1<=1'b0; have_r2<=1'b0; win_valid<=1'b0;
            w00<=0;w01<=0;w02<=0;w10<=0;w11<=0;w12<=0;w20<=0;w21<=0;w22<=0;
        end else if (in_valid) begin
            // shift left
            w00<=w01; w01<=w02;
            w10<=w11; w11<=w12;
            w20<=w21; w21<=w22;

            // bring new rightmost column
            w02 <= have_r2 ? lb1[col] : {CHW{1'b0}};
            w12 <= have_r1 ? lb0[col] : {CHW{1'b0}};
            w22 <= pix_in;

            // update line buffers
            lb1[col] <= lb0[col];
            lb0[col] <= pix_in;

            // column & row bookkeeping
            if (col==IMG_W-1) begin
                col <= 0;
                if (!have_r1) have_r1 <= 1'b1;
                else if (!have_r2) have_r2 <= 1'b1;
            end else begin
                col <= col + 1;
            end

            win_valid <= (have_r1 & have_r2) & (col>=2);
            if (win_valid) win_out <= {w00,w01,w02,w10,w11,w12,w20,w21,w22};
        end else begin
            win_valid <= 1'b0;
        end
    end
endmodule
