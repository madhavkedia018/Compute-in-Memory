module controller (
    input  clk, reset,
    input  start,
    output reg load_input,
    output reg load_weights,
    output reg compute,
    output reg done
);
    typedef enum logic [1:0] {IDLE, LOAD, COMPUTE, DONE} state_t;
    state_t state, next;

    always @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else       state <= next;
    end

    always @(*) begin
        load_input = 0;
        load_weights = 0;
        compute = 0;
        done = 0;
        case (state)
            IDLE:   next = start ? LOAD : IDLE;
            LOAD:   begin load_input = 1; load_weights = 1; next = COMPUTE; end
            COMPUTE: begin compute = 1; next = DONE; end
            DONE:   begin done = 1; next = IDLE; end
        endcase
    end
endmodule
