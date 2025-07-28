import numpy as np

def binary_conv2d(image, kernel, cim_ip):
    out = []
    for i in range(image.shape[0] - 2):
        row = []
        for j in range(image.shape[1] - 2):
            patch = image[i:i+3, j:j+3]
            patch_vec = int("".join(str(b) for b in patch.flatten()), 2)
            kernel_vec = int("".join(str(b) for b in kernel.flatten()), 2)
            cim_ip.write(0x10, patch_vec)
            cim_ip.write(0x18, kernel_vec)
            cim_ip.write(0x00, 0x01)
            while (cim_ip.read(0x00) & 0x2) == 0:
                pass
            result = cim_ip.read(0x20)
            row.append(result)
        out.append(row)
    return np.array(out)
