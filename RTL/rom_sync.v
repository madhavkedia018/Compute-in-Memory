// Synchronous ROM (inferred BRAM). One read per cycle.
// Initialize with $readmemb/.mem files exported by training.
module rom_sync #(
    parameter integer AW = 10,       // depth = 2^AW
    parameter integer DW = 256,
    parameter          INIT_FILE = "" // "weights.mem"
)(
    input  wire              clk,
    input  wire [AW-1:0]     addr,
    output reg  [DW-1:0]     dout
);
    (* rom_style="block" *) reg [DW-1:0] mem [0:(1<<AW)-1];
    initial if (INIT_FILE != "") $readmemb(INIT_FILE, mem);
    always @(posedge clk) dout <= mem[addr];
endmodule
