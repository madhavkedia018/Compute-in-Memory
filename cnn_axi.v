`timescale 1ns / 1ps

module cnn_axi #
(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6
)
(
    input wire clk,
    input wire resetn,

    // AXI Lite signals
    input  wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
    input  wire                            S_AXI_AWVALID,
    output wire                            S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
    input  wire                            S_AXI_WVALID,
    output wire                            S_AXI_WREADY,
    output wire [1 : 0]                    S_AXI_BRESP,
    output wire                            S_AXI_BVALID,
    input  wire                            S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
    input  wire                            S_AXI_ARVALID,
    output wire                            S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
    output wire [1 : 0]                    S_AXI_RRESP,
    output wire                            S_AXI_RVALID,
    input  wire                            S_AXI_RREADY
);

    // Register map
    // 0x00 : control (bit 0: start, bit 1: done)
    // 0x10 : input vector
    // 0x18 : weight vector
    // 0x20 : result

    reg start, done;
    reg [7:0] input_vec;
    reg [7:0] weight_vec;
    wire [7:0] result;

    // Connect to your CNN compute module
    cnn_top core (
        .clk(clk),
        .reset(~resetn),
        .start(start),
        .in_vector(input_vec),
        .weight_vector(weight_vec),
        .result(result),
        .done(done)
    );

    //---------------- AXI registers ----------------//
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg axi_awready = 0;
    reg axi_wready  = 0;
    reg axi_bvalid  = 0;
    reg [1:0] axi_bresp = 0;
    reg axi_arready = 0;
    reg axi_rvalid  = 0;
    reg [1:0] axi_rresp = 0;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata = 0;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    //---------------- Write Logic ----------------//
    always @(posedge clk) begin
        if (~resetn) begin
            axi_awready <= 0;
            axi_wready <= 0;
            axi_bvalid <= 0;
            start <= 0;
        end else begin
            if (S_AXI_AWVALID && !axi_awready) begin
                axi_awready <= 1;
                axi_awaddr <= S_AXI_AWADDR;
            end else begin
                axi_awready <= 0;
            end

            if (S_AXI_WVALID && !axi_wready) begin
                axi_wready <= 1;
                case (axi_awaddr)
                    6'h00: start <= S_AXI_WDATA[0];
                    6'h10: input_vec <= S_AXI_WDATA[7:0];
                    6'h18: weight_vec <= S_AXI_WDATA[7:0];
                endcase
            end else begin
                axi_wready <= 0;
            end

            if (axi_awready && axi_wready && !axi_bvalid) begin
                axi_bvalid <= 1;
                axi_bresp <= 2'b00; // OKAY
            end else if (S_AXI_BREADY && axi_bvalid) begin
                axi_bvalid <= 0;
            end
        end
    end

    //---------------- Read Logic ----------------//
    always @(posedge clk) begin
        if (~resetn) begin
            axi_arready <= 0;
            axi_rvalid <= 0;
            axi_rdata <= 0;
        end else begin
            if (S_AXI_ARVALID && !axi_arready) begin
                axi_arready <= 1;
                axi_araddr <= S_AXI_ARADDR;
            end else begin
                axi_arready <= 0;
            end

            if (axi_arready && !axi_rvalid) begin
                axi_rvalid <= 1;
                axi_rresp <= 2'b00; // OKAY
                case (axi_araddr)
                    6'h00: axi_rdata <= {30'd0, done, start};
                    6'h10: axi_rdata <= {24'd0, input_vec};
                    6'h18: axi_rdata <= {24'd0, weight_vec};
                    6'h20: axi_rdata <= {24'd0, result};
                    default: axi_rdata <= 32'hDEAD_BEEF;
                endcase
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 0;
            end
        end
    end

endmodule
