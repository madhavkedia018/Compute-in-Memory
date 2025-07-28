# ===================================================
# Pynq-Z2 Default Clock (125 MHz)
# ===================================================
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 8.000 -name sys_clk -waveform {0 4} [get_ports clk]

# ===================================================
# Active-Low Reset (Button0 on Pynq-Z2)
# ===================================================
set_property PACKAGE_PIN D19 [get_ports resetn]
set_property IOSTANDARD LVCMOS33 [get_ports resetn]

# ===================================================
# (Optional) Debug - LEDs for DONE and other signals
# ===================================================
# Uncomment these if you wish to use onboard LEDs for debugging
# set_property PACKAGE_PIN U16 [get_ports done]     # LD0
# set_property IOSTANDARD LVCMOS33 [get_ports done]

# set_property PACKAGE_PIN E19 [get_ports [result[0]]] # LD1
# set_property PACKAGE_PIN U19 [get_ports [result[1]]] # LD2
# ...

# You can also use PMOD header (JA, JB, etc.) for GPIO output
# set_property PACKAGE_PIN J1 [get_ports gpio_out[0]]
# set_property IOSTANDARD LVCMOS33 [get_ports gpio_out[0]]
