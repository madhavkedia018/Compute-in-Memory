module top_vgg16_bnn_pynq #(
    parameter integer IMG_W   = 224,
    parameter integer CH_IN   = 64,   // interleaved input channels (binary)
    parameter integer CH_MID  = 64,
    parameter integer CH_OUT  = 64,
    parameter integer SIMD    = 64,
    parameter integer PE      = 8,
    parameter integer SUMW    = 16
)(
    input  wire                clk125,     // 125 MHz PL clock (H16)
    input  wire                rst_n,      // active-low reset (BTN0)
    
    
    input  wire                in_valid,
    input  wire [CH_IN-1:0]    in_pix,     // binary channels interleaved per pixel
    output wire                out_valid,
    output wire [9:0]          out_bits,   // example 10-way classification
    
      // AXI-controlled control/addresses
//  input  wire        start,
//  output reg         done,
//  input  wire [31:0] in_base_addr,
//  input  wire [31:0] out_base_addr,

//  // BRAM handshake (simple)
//  input  wire [31:0] bram_din,
//  output reg  [31:0] bram_addr,
//  output reg         bram_rd_en,
//  output reg  [31:0] bram_dout,
//  output reg  [31:0] bram_wr_addr,
//  output reg         bram_wr_en,
    
    output wire [3:0] leds        // heartbeat/debug
);
    
    
    
    // internal state
localparam IDLE=0, READ=1, PROCESS=2, WRITE=3, DONE=4;
reg [2:0] state;
reg [31:0] read_ptr; // word offset
reg [31:0] write_ptr;

//always @(posedge clk125) begin
//  if (!rst_n) begin
//    state <= IDLE;
//    done <= 0;
//    bram_rd_en <= 0;
//    bram_wr_en <= 0;
//    read_ptr <= 0;
//    write_ptr <= 0;
//  end else begin
//    case (state)
//      IDLE: begin
//        done <= 0;
//        if (start) begin
//          read_ptr <= in_base_addr;
//          write_ptr <= out_base_addr;
//          state <= READ;
//        end
//      end
//      READ: begin
//        // request BRAM read
//        bram_addr <= read_ptr;
//        bram_rd_en <= 1'b1;
//        state <= PROCESS; // on next cycle bram_din contains data
//      end
//      PROCESS: begin
//        bram_rd_en <= 1'b0;
//        // Convert bram_din -> in_pix & assert in_valid for pipeline
//        // (YOU MUST ADAPT: packing/width specifics depend on your pixel layout)
//        // e.g: in_pix <= bram_din[CH_IN-1:0]; in_valid <= 1;
//        // After feeding all pixels (loop), move to WRITE
//        // else increment read_ptr and go to READ
//      end
//      WRITE: begin
//        // After pipeline finishes, write results to BRAM at write_ptr
//        bram_dout <= {22'd0, out_bits}; // pack output bits into 32-bit word
//        bram_wr_addr <= write_ptr;
//        bram_wr_en <= 1'b1;
//        // single word write, then state DONE
//        state <= DONE;
//      end
//      DONE: begin
//        bram_wr_en <= 1'b0;
//        done <= 1;
//        // wait for PS to clear start (or a new control)
//        if (!start) state <= IDLE;
//      end
//    endcase
//  end
//end

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    wire rst = ~rst_n;
//    wire [9:0] out_bits;
//    wire  out_valid;
//    wire in_valid;
//    wire [CH_IN-1:0] in_pix;
    
    // --- Block 1: SWU -> MVTU (3x3 conv) -> Pool ---
    wire                w1_valid;
    wire [9*CH_IN-1:0]  w1;
    swu_3x3 #(.IMG_W(IMG_W), .CHW(CH_IN)) SWU1 (
        .clk(clk125), .rst(rst), .in_valid(in_valid), .pix_in(in_pix),
        .win_valid(w1_valid), .win_out(w1)
    );

    // Extract SIMD slice to feed MVTU (here we reuse MVTU as "conv by GEMM")
    // In practice you'd stream columns of the lowered image; for brevity we feed w1[SIMD-1:0]
    wire                c1_valid;
    wire [CH_MID-1:0]   c1_bits;

    mvtu #(
        .INW (9*CH_IN),
        .OUT (CH_MID),
        .PE  (PE),
        .SIMD(SIMD),
        .SUMW(SUMW),
        .W_AW(10), .T_AW(7),
        .W_INIT("c1_weights.mem"),
        .T_INIT("c1_thresh.mem")
    ) C1 (
        .clk(clk125), .rst(rst),
        .in_valid(w1_valid),
        .in_vec(w1),          // full 3x3xCin vector
        .out_valid(c1_valid),
        .out_bits(c1_bits)
    );

    wire                p1_valid;
    wire [CH_MID-1:0]   p1_bits;
    pool2x2_bin #(.IMG_W(IMG_W), .CHW(CH_MID)) P1 (
        .clk(clk125), .rst(rst),
        .in_valid(c1_valid), .pix_in(c1_bits),
        .out_valid(p1_valid), .pix_out(p1_bits)
    );

    // --- Block 2: SWU (now IMG_W/2) -> MVTU ---
    wire                w2_valid;
    wire [9*CH_MID-1:0] w2;
    swu_3x3 #(.IMG_W(IMG_W/2), .CHW(CH_MID)) SWU2 (
        .clk(clk125), .rst(rst), .in_valid(p1_valid), .pix_in(p1_bits),
        .win_valid(w2_valid), .win_out(w2)
    );

    wire                c2_valid;
    wire [CH_OUT-1:0]   c2_bits;
    mvtu #(
        .INW (9*CH_MID),
        .OUT (CH_OUT),
        .PE  (PE),
        .SIMD(SIMD),
        .SUMW(SUMW),
        .W_AW(10), .T_AW(7),
        .W_INIT("c2_weights.mem"),
        .T_INIT("c2_thresh.mem")
    ) C2 (
        .clk(clk125), .rst(rst),
        .in_valid(w2_valid),
        .in_vec(w2),
        .out_valid(c2_valid),
        .out_bits(c2_bits)
    );

    // --- Tiny FC head (flatten stub): reuse MVTU with INW=CH_OUT, OUT=10 ---
    wire               fc_valid;
    mvtu #(
        .INW (CH_OUT),
        .OUT (10),
        .PE  (5),        // example folding
        .SIMD(32),       // example SIMD
        .SUMW(8),
        .W_AW(8), .T_AW(4),
        .W_INIT("fc_weights.mem"),
        .T_INIT("fc_thresh.mem")
    ) FC0 (
        .clk(clk125), .rst(rst),
        .in_valid(c2_valid),
        .in_vec({{(32-CH_OUT){1'b0}}, c2_bits}), // zero-extend to SIMD boundary if needed
        .out_valid(fc_valid),
        .out_bits(out_bits)
    );

    assign out_valid = fc_valid;

    // Heartbeat/debug LEDs
    reg [23:0] hb;
    always @(posedge clk125) if (rst) hb<=0; else hb<=hb+1;
    assign leds = {out_valid, hb[23], hb[22], hb[21]};
endmodule
