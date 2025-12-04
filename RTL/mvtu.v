// mvtu.v - FINN-style MVTU (binary) - Vivado-safe, single writer for out_bits

module mvtu #(
    parameter integer INW  = 576,
    parameter integer OUT  = 64,
    parameter integer PE   = 8,
    parameter integer SIMD = 64,
    parameter integer SUMW = 16,
    parameter integer W_AW = 10,
    parameter integer T_AW = 6,
    parameter        W_INIT = "",
    parameter        T_INIT = ""
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   in_valid,
    input  wire [INW-1:0]         in_vec,
    output reg                    out_valid,
    output reg  [OUT-1:0]         out_bits
);


//wire                   in_valid;
//wire [INW-1:0]         in_vec;
//reg                    out_valid;
//reg  [OUT-1:0]         out_bits;
    // ROMs
    wire [SIMD-1:0] w_dout [0:PE-1];
    wire [SUMW-1:0] t_dout [0:PE-1];
    reg  [W_AW-1:0] w_addr [0:PE-1];
    reg  [T_AW-1:0] t_addr [0:PE-1];

    genvar p;
    generate
        for (p=0; p<PE; p=p+1) begin: PEW
            rom_sync #(.AW(W_AW), .DW(SIMD), .INIT_FILE(W_INIT)) WROM (
                .clk(clk), .addr(w_addr[p]), .dout(w_dout[p])
            );
            rom_sync #(.AW(T_AW), .DW(SUMW), .INIT_FILE(T_INIT)) TROM (
                .clk(clk), .addr(t_addr[p]), .dout(t_dout[p])
            );
        end
    endgenerate

    // Counters/accums
    localparam integer COL_TILES = (INW + SIMD - 1) / SIMD;
    reg [$clog2(COL_TILES):0] col_cnt;
    reg [$clog2(OUT/PE):0]    row_blk;
    reg [SUMW-1:0]            acc [0:PE-1];
    integer i;

    // XNOR+popcount lanes
    wire [$clog2(SIMD+1)-1:0] pcnt [0:PE-1];
    generate
        for (p=0; p<PE; p=p+1) begin: XPC
            xnor_popcount #(.W(SIMD)) XPC_I (
                .a(in_vec[SIMD-1:0]),
                .b(w_dout[p]),
                .sum(pcnt[p])
            );
        end
    endgenerate

    // Flattened threshold staging
    reg [PE*SUMW-1:0] thresh_sum;
    reg [PE*SUMW-1:0] thresh_thr;
    wire [PE-1:0]     act_bits_local;

    (* keep_hierarchy = "yes" *)
    thresh_act #(.NCH(PE), .SUMW(SUMW)) ACT1 (
        .sum_in  (thresh_sum),
        .thresh  (thresh_thr),
        .act_bits(act_bits_local)
    );

    // Single writer for out_bits + control
    always @(posedge clk) begin
        if (rst) begin
            col_cnt    <= 0;
            row_blk    <= 0;
            out_valid  <= 1'b0;
            out_bits   <= {OUT{1'b0}};
            thresh_sum <= {PE*SUMW{1'b0}};
            thresh_thr <= {PE*SUMW{1'b0}};
            for (i=0;i<PE;i=i+1) begin
                acc[i]    <= {SUMW{1'b0}};
                w_addr[i] <= {W_AW{1'b0}};
                t_addr[i] <= {T_AW{1'b0}};
            end
        end else begin
            out_valid <= 1'b0;

            if (in_valid) begin
                // accumulate current SIMD tile
                for (i=0;i<PE;i=i+1) begin
                    acc[i]    <= (col_cnt==0) ? {{(SUMW-$clog2(SIMD+1)){1'b0}}, pcnt[i]} :
                                              (acc[i] + {{(SUMW-$clog2(SIMD+1)){1'b0}}, pcnt[i]});
                    w_addr[i] <= w_addr[i] + 1;
                end

                if (col_cnt == COL_TILES-1) begin
                    // stage sums/thresholds and compute activations next cycle
                    for (i=0;i<PE;i=i+1) begin
                        thresh_sum[i*SUMW +: SUMW] <= acc[i];
                        thresh_thr[i*SUMW +: SUMW] <= t_dout[i];
                        t_addr[i] <= t_addr[i] + 1;
                    end
                    out_valid <= 1'b1;
                    // write out_bits at same clock edge using last stage activations
                    for (i=0;i<PE;i=i+1)
                        out_bits[row_blk*PE + i] <= act_bits_local[i];

                    row_blk <= (row_blk == (OUT/PE)-1) ? 0 : (row_blk+1);
                    col_cnt <= 0;
                end else begin
                    col_cnt <= col_cnt + 1;
                end
            end
        end
    end

endmodule
