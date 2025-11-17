import numpy as np
from PIL import Image

# ============================================================
#  SETTINGS
# ============================================================
IMG_W = 32
CH_IN = 64              # Your top module uses CH_IN=64 binary channels
OUT_FILE = "input_pixels.mem"

# ============================================================
# 1. Load CIFAR-10 image (or any 32x32 image)
# ============================================================
# Use your own file here
img_path = "cat.jpg"

img = Image.open(img_path).convert("RGB")
img = img.resize((IMG_W, IMG_W), Image.BILINEAR)
img_np = np.asarray(img)   # shape = (32,32,3)
img_np = img_np.astype(np.float32) / 255.0   # normalize to 0..1


# ============================================================
# 2. Convert RGB to 64 binary feature channels
# ============================================================
# NOTE:
# For a REAL BNN (like FINN CNV-W1A1) this stage is normally:
#    Conv / BatchNorm / Sign activation
# However you do not have those parameters yet.
#
# So here we generate a "synthetic" feature map:
#   64 binary channels = thresholded linear combinations of R,G,B
# This is ONLY for testing your pipeline timing & dataflow.
# Later we replace this with real FINN preprocessing.
# ============================================================

def generate_64_binary_channels(pixel):
    """Generate 64 binary channels from a 3-channel pixel.
       This is synthetic but stable for testing the hardware pipeline."""

    r, g, b = pixel

    # simple features
    feats = [
        r > 0.5,
        g > 0.5,
        b > 0.5,
        r > g,
        g > b,
        b > r,
        (r + g + b) > 1.0,
        (r + g) > (b + 0.2),
    ]

    # Expand to 64 features by repeating with noise thresholds
    feats64 = []
    for i in range(CH_IN):
        thresh = (i % 10) / 10.0  # threshold pattern
        c = pixel[i % 3]          # pick R,G, or B
        feats64.append(c > thresh)

    return feats64


# ============================================================
# 3. Pack each pixel's 64 binary features into a 64-bit int
# ============================================================
packed_pixels = []

for y in range(IMG_W):
    for x in range(IMG_W):
        pixel = img_np[y, x, :]   # [r,g,b]
        feats = generate_64_binary_channels(pixel)  # list of 64 booleans

        # pack MSB..LSB → bit63..bit0
        w = 0
        for i, bit in enumerate(feats):
            if bit:
                w |= (1 << (63 - i))   # MSB aligned

        packed_pixels.append(w)


# ============================================================
# 4. Write output as .mem file
# ============================================================
with open(OUT_FILE, "w") as f:
    for w in packed_pixels:
        f.write(f"{w:016x}\n")    # 64-bit hex word per line

print(f"Generated {OUT_FILE} with {len(packed_pixels)} pixels.")
