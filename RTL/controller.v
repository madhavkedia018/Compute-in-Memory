module controller (
    input  clk, reset,
    input  start,
    output reg load_input,
    output reg load_weights,
    output reg compute,
    output reg done
);
    // State encoding
    parameter IDLE    = 2'b00;
    parameter LOAD    = 2'b01;
    parameter COMPUTE = 2'b10;
    parameter DONE    = 2'b11;

    reg [1:0] state, next;

    // Sequential state update
    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= IDLE;
        else
            state <= next;
    end

    // Combinational next-state logic + outputs
    always @(*) begin
        load_input  = 0;
        load_weights= 0;
        compute     = 0;
        done        = 0;
        case (state)
            IDLE:    next = start ? LOAD : IDLE;
            LOAD:    begin
                        load_input   = 1;
                        load_weights = 1;
                        next = COMPUTE;
                     end
            COMPUTE: begin
                        compute = 1;
                        next = DONE;
                     end
            DONE:    begin
                        done = 1;
                        next = IDLE;
                     end
            default: next = IDLE;
        endcase
    end
endmodule
