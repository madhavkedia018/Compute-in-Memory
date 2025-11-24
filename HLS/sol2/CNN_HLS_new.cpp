#include "ap_int.h"
#include <math.h>

// SAME SHAPES AS BNN VERSION
#define IMG_H 32
#define IMG_W 32
#define K 3
#define POOL 2

#define C0_IN 3
#define C0_OUT 128   // (not used below, but kept for clarity)

#define C1_IN 128
#define C1_OUT 128

#define C2_IN 128
#define C2_OUT 256

#define C3_IN 256
#define C3_OUT 256

#define C4_IN 256
#define C4_OUT 512

#define C5_IN 512
#define C5_OUT 512

#define NUM_CLASSES 10

#define H0 IMG_H
#define W0 IMG_W

#define H1 (H0/POOL)
#define W1 (W0/POOL)

#define H2 H1
#define W2 W1

#define H3 (H2/POOL)
#define W3 (W2/POOL)

#define H4 H3
#define W4 W3

#define H5 (H4/POOL)
#define W5 (W4/POOL)

// ====== FLOAT WEIGHTS (Dummy) ==================================================
static float F_W1[C1_OUT][C0_IN][K][K];   // <-- FIXED: use C0_IN here
static float F_B1[C1_OUT];

static float F_W2[C2_OUT][C2_IN][K][K];
static float F_B2[C2_OUT];

static float F_W3[C3_OUT][C3_IN][K][K];
static float F_B3[C3_OUT];

static float F_W4[C4_OUT][C4_IN][K][K];
static float F_B4[C4_OUT];

static float F_W5[C5_OUT][C5_IN][K][K];
static float F_B5[C5_OUT];

static float FC_WF[NUM_CLASSES][C5_OUT*H5*W5];
static float FC_BF[NUM_CLASSES];


// ===== FLOAT CONVOLUTION ========================================================
template<int IN_C, int OUT_C, int IN_H, int IN_W, int OUT_H, int OUT_W>
void float_conv(float in[IN_C][IN_H][IN_W],
                float out[OUT_C][OUT_H][OUT_W],
                float W[OUT_C][IN_C][K][K],
                float B[OUT_C])
{
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oy = 0; oy < OUT_H; oy++) {
            for (int ox = 0; ox < OUT_W; ox++) {
#pragma HLS PIPELINE II=1
                float acc = B[oc];
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            acc += in[ic][oy + ky][ox + kx] *
                                   W[oc][ic][ky][kx];
                        }
                    }
                }
                out[oc][oy][ox] = acc;
            }
        }
    }
}

// ===== MaxPool =================================================================
template<int C, int IN_H, int IN_W, int OUT_H, int OUT_W>
void maxpool(float in[C][IN_H][IN_W], float out[C][OUT_H][OUT_W])
{
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < OUT_H; y++) {
            for (int x = 0; x < OUT_W; x++) {
#pragma HLS PIPELINE II=1
                float m = -1e9;
                for (int ky = 0; ky < POOL; ky++)
                    for (int kx = 0; kx < POOL; kx++) {
                        float v = in[c][y * POOL + ky][x * POOL + kx];
                        if (v > m) m = v;
                    }
                out[c][y][x] = m;
            }
        }
    }
}

// ===== Hardtanh ================================================================
template<int C, int H, int W>
void hardtanh(float x[C][H][W])
{
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x1 = 0; x1 < W; x1++) {
#pragma HLS PIPELINE II=1
                float v = x[c][y][x1];
                if (v > 1) v = 1;
                if (v < -1) v = -1;
                x[c][y][x1] = v;
            }
}

// ===== Fully Connected =========================================================
void fc_float(float in[C5_OUT][H5][W5], float out[NUM_CLASSES])
{
    float flat[C5_OUT * H5 * W5];
    int idx = 0;

    for (int c = 0; c < C5_OUT; c++)
        for (int y = 0; y < H5; y++)
            for (int x = 0; x < W5; x++)
                flat[idx++] = in[c][y][x];

    for (int o = 0; o < NUM_CLASSES; o++) {
        float acc = FC_BF[o];
        for (int i = 0; i < C5_OUT * H5 * W5; i++)
            acc += flat[i] * FC_WF[o][i];
        out[o] = acc;
    }
}


// ================= TOP ============================
extern "C" void CNN_VGG_FORWARD(
    float input[C0_IN][IMG_H][IMG_W],
    float output[NUM_CLASSES])
{
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=output bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    static float F1[C1_OUT][H1][W1];
    static float F2[C2_OUT][H2][W2];
    static float F3[C3_OUT][H3][W3];
    static float F4[C4_OUT][H4][W4];
    static float F5[C5_OUT][H5][W5];

    // Conv 1  (FIXED: use C0_IN as IN_C)
    float_conv<C0_IN, C1_OUT, H0, W0, H1, W1>(input, F1, F_W1, F_B1);
    maxpool<C1_OUT, H1, W1, H1, W1>(F1, F1);
    hardtanh<C1_OUT, H1, W1>(F1);

    // Conv 2
    float_conv<C2_IN, C2_OUT, H1, W1, H2, W2>(F1, F2, F_W2, F_B2);
    hardtanh<C2_OUT, H2, W2>(F2);

    // Conv 3
    float_conv<C3_IN, C3_OUT, H2, W2, H3, W3>(F2, F3, F_W3, F_B3);
    maxpool<C3_OUT, H3, W3, H3, W3>(F3, F3);
    hardtanh<C3_OUT, H3, W3>(F3);

    // Conv 4
    float_conv<C4_IN, C4_OUT, H3, W3, H4, W4>(F3, F4, F_W4, F_B4);
    hardtanh<C4_OUT, H4, W4>(F4);

    // Conv 5
    float_conv<C5_IN, C5_OUT, H4, W4, H5, W5>(F4, F5, F_W5, F_B5);
    maxpool<C5_OUT, H5, W5, H5, W5>(F5, F5);
    hardtanh<C5_OUT, H5, W5>(F5);

    // FC
    fc_float(F5, output);
}
