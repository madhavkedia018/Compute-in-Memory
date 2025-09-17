`timescale 1ns/1ps

module tb_cim_memory;

    // Parameters
    parameter N = 8;

    // Testbench signals
    reg  [N-1:0] input_vector;
    reg  [N-1:0] weight_vector;
    wire signed [7:0] dot_val;

    // DUT instantiation
    cim_memory #(N) dut (
        .input_vector(input_vector),
        .weight_vector(weight_vector),
        .dot_val(dot_val)
    );

    initial begin
        $display("===== CIM Memory Testbench Start =====");

        // Test 1: All zeros
        input_vector = 8'b00000000;
        weight_vector = 8'b00000000;
        #10;
        $display("input=%b weight=%b dot_val=%0d", input_vector, weight_vector, dot_val);

        // Test 2: All ones
        input_vector = 8'b11111111;
        weight_vector = 8'b11111111;
        #10;
        $display("input=%b weight=%b dot_val=%0d", input_vector, weight_vector, dot_val);

        // Test 3: Opposites
        input_vector = 8'b11111111;
        weight_vector = 8'b00000000;
        #10;
        $display("input=%b weight=%b dot_val=%0d", input_vector, weight_vector, dot_val);

        // Test 4: Half match
        input_vector = 8'b11110000;
        weight_vector = 8'b11001100;
        #10;
        $display("input=%b weight=%b dot_val=%0d", input_vector, weight_vector, dot_val);

        // Test 5: Random 1
        input_vector = 8'b10101010;
        weight_vector = 8'b11001100;
        #10;
        $display("input=%b weight=%b dot_val=%0d", input_vector, weight_vector, dot_val);

        // Test 6: Random 2
        input_vector = 8'b01101001;
        weight_vector = 8'b10110110;
        #10;
        $display("input=%b weight=%b dot_val=%0d", input_vector, weight_vector, dot_val);

        $display("===== CIM Memory Testbench End =====");
        $finish;
    end

endmodule
