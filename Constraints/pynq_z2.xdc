## PYNQ-Z2 (XC7Z020-1CLG400C) minimal master XDC
## Clock: 125 MHz external PL clock on H16
set_property -dict { PACKAGE_PIN H16  IOSTANDARD LVCMOS33 } [get_ports clk125]
create_clock -add -name clk125 -period 8.000 [get_ports clk125]

## Buttons (active-high on board; we invert in RTL if we want active-low)
## Here we use BTN0 as active-low reset_n
set_property -dict { PACKAGE_PIN D19 IOSTANDARD LVCMOS33 } [get_ports rst_n]

## LEDs (4 user LEDs)
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports { leds[0] }]
set_property -dict { PACKAGE_PIN M15 IOSTANDARD LVCMOS33 } [get_ports { leds[1] }]
set_property -dict { PACKAGE_PIN G14 IOSTANDARD LVCMOS33 } [get_ports { leds[2] }]
set_property -dict { PACKAGE_PIN D18 IOSTANDARD LVCMOS33 } [get_ports { leds[3] }]

## (Optional) Switches
# set_property -dict { PACKAGE_PIN G15 IOSTANDARD LVCMOS33 } [get_ports { sw[0] }]
# set_property -dict { PACKAGE_PIN P15 IOSTANDARD LVCMOS33 } [get_ports { sw[1] }]
