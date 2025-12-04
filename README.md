# ⚡ Energy-Efficient BNN Accelerator on FPGA  
### Bit-Level CIM Dataflow | Approximate Adders | PYNQ-Z2 FPGA | CIFAR-10

---

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-PYNQ--Z2-blue.svg)
![Language](https://img.shields.io/badge/Language-Python%20%7C%20Verilog-orange.svg)

---

## 📌 Project Overview

This project implements an **energy-efficient Binary Neural Network (BNN) accelerator** on the **PYNQ-Z2 FPGA**, using:

- **1-bit weights and activations (1W1A)**
- **Bit-level compute-in-memory (CIM) dataflow**
- **LUT-based XNOR–POPCOUNT convolution units**
- **Approximate adders for low-power accumulation**
- **BRAM-mapped binary weights and thresholds**

The accelerator is integrated into the PYNQ framework as a **custom hardware overlay**, compared against a software CNN baseline running on the ARM Cortex-A9.

---

## 🚀 Features

- FPGA-based BNN accelerator with microsecond-level latency  
- Approximate computing for reduced power and hardware usage  
- End-to-end pipeline: training → binarisation → RTL → bitstream  
- Live inference using Webcam + PYNQ Jupyter Notebook  
- CNN vs BNN comparison (accuracy, latency, power, memory)

---


## ⚙️ System Architecture

### 🔹 End-to-End Workflow
1. Train full-precision VGG model on CIFAR-10  
2. Apply **IR-Net (Information Retention)** based binarisation  
3. Export **1-bit weights & thresholds** to C headers  
4. Implement hardware blocks in Verilog (SWU, MVU, Pooling, FC)  
5. Synthesize in Vivado → generate `bnn.bit` & `bnn.hwh`  
6. Load overlay on PYNQ → run inference via Python API  

---

## 📊 Results

### ✔ Accuracy

| Model      | Accuracy |
|------------|----------|
| **CNN (CPU)** | ~85%     |
| **BNN (FPGA)** | ~80%     |

---

### ✔ Latency

| Model      | Latency per Image |
|------------|-------------------|
| **CNN (CPU)** | 1.5–2 s           |
| **BNN (FPGA)** | 1.5–2 ms          |

---

### ✔ Power Usage (Vivado Report)

- **Total On-Chip Power:** 0.11 W  
- **Dynamic Power:** 0.004 W  
- **DSP Usage:** 0  
- **LUT Usage:** Very Low (LUT-only design)

---

### ✔ Real-Time Webcam Demo

BNN correctly classifies objects live from a webcam:


 <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/e57fdee8-7cb8-48d2-b79a-e268bde830b8" />  <img width="500" height="700" alt="image" src="https://github.com/user-attachments/assets/a48889c4-ead3-4bd4-a80d-44218eadb781" />


---

## 📘 Documentation

The complete Major Project Report is available here: https://drive.google.com/file/d/1zeQQKxoYLFZR4dIEqhLo2raIPfjUwMdd/view?usp=drive_link


---

## 🛠 Future Work

-[ ] Add multi-bit activations (2–4 bit XNOR networks)  
-[ ] Extend to larger datasets (Tiny-ImageNet / ImageNet-100)  
-[ ] Integrate DMA-based streaming for higher throughput  
-[ ] Deploy on more advanced FPGA boards (ZCU104 / Ultra96-V2)




