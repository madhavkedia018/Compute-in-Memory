import os
import numpy as np

# Path where the *.bin files live (YOUR PATH)
PARAM_DIR = r"C:/FPGA/BNN-PYNQ-master/BNN-PYNQ-master/bnn/params/cifar10/cnvW1A1"

# Where to write the .mem files (can be your Vivado project folder)
OUT_DIR = os.path.join(PARAM_DIR, "mem_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Width of ROM words (must match your Verilog)
SIMD = 64
THRESH_WIDTH = 16  # SUMW in your design

def load_bin(path):
    """Load FINN bit-packed file as a flat bit array."""
    data = np.fromfile(path, dtype=np.uint8)
    return np.unpackbits(data)

def pack_bits(bits, width):
    lines = []
    for i in range(0, len(bits), width):
        chunk = bits[i:i+width]
        if len(chunk) < width:
            chunk = np.pad(chunk, (0, width - len(chunk)))
        line = ''.join(str(b) for b in chunk)
        lines.append(line)
    return lines

def write_mem(filename, lines):
    with open(filename, "w") as f:
        for l in lines:
            f.write(l + "\n")
    print("WROTE:", filename, "(", len(lines), "lines )")

def convert_layer(layer_idx, name):
    print(f"\nConverting Layer {layer_idx} → {name}")
    weight_lines = []
    thresh_lines = []

    # FINN usually uses PE = 8 for CNV W1A1
    for pe in range(16):  # check up to 16, break when PE files stop existing
        wfile = os.path.join(PARAM_DIR, f"{layer_idx}-{pe}-weights.bin")
        tfile = os.path.join(PARAM_DIR, f"{layer_idx}-{pe}-thres.bin")

        if not os.path.exists(wfile):
            break

        print("  Reading:", wfile)

        W = load_bin(wfile)
        T = load_bin(tfile)

        weight_lines += pack_bits(W, SIMD)
        thresh_lines += pack_bits(T, THRESH_WIDTH)

    write_mem(os.path.join(OUT_DIR, f"{name}_weights.mem"), weight_lines)
    write_mem(os.path.join(OUT_DIR, f"{name}_thresh.mem"), thresh_lines)


print("\n=== Converting FINN CNV-W1A1 Parameters ===\n")

# Conv layer 1
convert_layer(0, "c1")

# Conv layer 2
convert_layer(1, "c2")

# FC layer (layer index = 3 in FINN)
convert_layer(3, "fc")

print("\nDone! .mem files saved to:", OUT_DIR)
