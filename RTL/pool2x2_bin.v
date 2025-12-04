module pool2x2_bin #(
    parameter integer IMG_W = 224,
    parameter integer CHW   = 64
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    input  wire [CHW-1:0]    pix_in,
    output reg                out_valid,
    output reg  [CHW-1:0]    pix_out
);
    reg [$clog2(IMG_W):0] col;
    reg row_odd, col_odd;
    reg [CHW-1:0] tl,tr,bl;

    always @(posedge clk) begin
        if (rst) begin
            col<=0; row_odd<=0; col_odd<=0; out_valid<=1'b0; pix_out<=0;
        end else if (in_valid) begin
            col_odd <= ~col_odd;
            if (col==IMG_W-1) begin col<=0; col_odd<=1'b0; row_odd<=~row_odd; end
            else col <= col + 1;

            if (!row_odd && !col_odd) tl <= pix_in;
            else if (!row_odd && col_odd) tr <= pix_in;
            else if ( row_odd && !col_odd) bl <= pix_in;
            else begin
                pix_out   <= tl | tr | bl | pix_in;
                out_valid <= 1'b1;
            end
            if (!(row_odd & col_odd)) out_valid <= 1'b0;
        end else out_valid <= 1'b0;
    end
endmodule
