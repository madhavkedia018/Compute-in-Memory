module cnn_top (
    input clk, reset, start,
    input [7:0] in_vector,
    input [7:0] weight_vector,
    output [7:0] result,
    output done
);
    wire [$clog2(9)-1:0] cim_out;
    wire [7:0] mult_out, sum_out;
    wire load_input, load_weights, compute;

    cim_memory #(8) mem_inst (
        .input_vector(in_vector),
        .weight_vector(weight_vector),
        .popcount_out(cim_out)
    );

    approx_mult mult_inst (
        .A({4'b0000, cim_out}),  // Extend to 4 bits
        .B(4'd3),                // Example scaling factor
        .P(mult_out)
    );

    approx_add add_inst (
        .A(mult_out),
        .B(8'd10),
        .SUM(sum_out)
    );

    controller ctrl (
        .clk(clk), .reset(reset), .start(start),
        .load_input(load_input),
        .load_weights(load_weights),
        .compute(compute),
        .done(done)
    );

    assign result = sum_out;
endmodule
