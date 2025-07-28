from pynq import Overlay
from pynq import allocate
import numpy as np
import time

# Load your bitstream
overlay = Overlay('/home/xilinx/cim_cnn.bit')
cim_ip = overlay.cnn_top_0  # Your IP block name

# Example: 3x3 kernel and 3x3 image
image = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1]
], dtype=np.uint8)

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=np.uint8)

# Flatten and bitpack if needed
image_vec = int("".join(str(b) for b in image.flatten()), 2)
kernel_vec = int("".join(str(b) for b in kernel.flatten()), 2)

# Send inputs to the IP (use memory-mapped registers)
cim_ip.write(0x10, image_vec)     # Assume offset 0x10 = input vector
cim_ip.write(0x18, kernel_vec)    # Assume offset 0x18 = weight vector
cim_ip.write(0x00, 0x01)          # Start the computation

# Wait for done flag (polling)
while (cim_ip.read(0x00) & 0x2) == 0:
    pass

# Read the result
result = cim_ip.read(0x20)  # Assume result stored at 0x20
print("Output:", result)
