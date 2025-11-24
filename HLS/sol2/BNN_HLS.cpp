#include "ap_int.h"
#include <math.h>

#define IMG_H 32
#define IMG_W 32
#define K 3
#define POOL 2

// ====== CHANNEL SIZES =======
#define C0_IN   3
#define C0_OUT 128

#define C1_IN  128
#define C1_OUT 128

#define C2_IN  128
#define C2_OUT 256

#define C3_IN  256
#define C3_OUT 256

#define C4_IN  256
#define C4_OUT 512

#define C5_IN  512
#define C5_OUT 512

#define NUM_CLASSES 10

// ====== FEATURE MAP SIZES ACROSS NETWORK ======

#define H0 IMG_H       // after conv0
#define W0 IMG_W

#define H1 (H0/POOL)   // after conv1 + pool
#define W1 (W0/POOL)

#define H2 (H1)        // after conv2 only
#define W2 (W1)

#define H3 (H2/POOL)   // after conv3 + pool
#define W3 (W2/POOL)

#define H4 (H3)        // after conv4 only
#define W4 (W3)

#define H5 (H4/POOL)   // after conv5 + pool
#define W5 (W4/POOL)

// ====== WEIGHTS (Dummy) ======
static signed char W1[C1_OUT][C1_IN][K][K];
static float SCALE1[C1_OUT];
static float BIAS1[C1_OUT];

static signed char W2[C2_OUT][C2_IN][K][K];
static float SCALE2[C2_OUT];
static float BIAS2[C2_OUT];

static signed char W3[C3_OUT][C3_IN][K][K];
static float SCALE3[C3_OUT];
static float BIAS3[C3_OUT];

static signed char W4[C4_OUT][C4_IN][K][K];
static float SCALE4[C4_OUT];
static float BIAS4[C4_OUT];

static signed char W5[C5_OUT][C5_IN][K][K];
static float SCALE5[C5_OUT];
static float BIAS5[C5_OUT];

static float FC_W[NUM_CLASSES][C5_OUT*H5*W5];
static float FC_B[NUM_CLASSES];

// Helper: sign → bit
inline int signbitf(float x) { return x >= 0 ? 1 : 0; }

// XNOR Binary Convolution (single layer)
template<int IN_C, int OUT_C, int IN_H, int IN_W, int OUT_H, int OUT_W>
void xnor_conv(float in[IN_C][IN_H][IN_W],
               float out[OUT_C][OUT_H][OUT_W],
               signed char W[OUT_C][IN_C][K][K],
               float SCALE[OUT_C],
               float BIAS[OUT_C])
{
    conv_loop:
    for(int oc=0; oc<OUT_C; oc++){
        for(int oy=0; oy<OUT_H; oy++){
            for(int ox=0; ox<OUT_W; ox++){
#pragma HLS PIPELINE II=1
                int pop = 0;
                int total = IN_C*K*K;

                for(int ic=0; ic<IN_C; ic++){
                    for(int ky=0; ky<K; ky++){
                        for(int kx=0; kx<K; kx++){
                            float a = in[ic][oy+ky][ox+kx];
                            int abit = signbitf(a);
                            int wbit = (W[oc][ic][ky][kx] > 0) ? 1 : 0;
                            pop += (abit == wbit);
                        }
                    }
                }
                int dot = 2*pop - total;     // popcount → dot-product
                out[oc][oy][ox] = SCALE[oc]*dot + BIAS[oc];
            }
        }
    }
}

// MaxPool 2×2
template<int C, int IN_H, int IN_W, int OUT_H, int OUT_W>
void maxpool(float in[C][IN_H][IN_W],
             float out[C][OUT_H][OUT_W])
{
    for(int c=0;c<C;c++){
        for(int y=0;y<OUT_H;y++){
            for(int x=0;x<OUT_W;x++){
#pragma HLS PIPELINE II=1
                float m = -1e9;
                for(int ky=0;ky<POOL;ky++)
                    for(int kx=0;kx<POOL;kx++){
                        float v = in[c][y*POOL+ky][x*POOL+kx];
                        if(v > m) m = v;
                    }
                out[c][y][x] = m;
            }
        }
    }
}

// Hardtanh Activation
template<int C, int H, int W>
void hardtanh(float x[C][H][W])
{
    for(int c=0;c<C;c++)
        for(int y=0;y<H;y++)
            for(int x1=0;x1<W;x1++){
#pragma HLS PIPELINE II=1
                float v = x[c][y][x1];
                if(v>1) v=1;
                if(v<-1) v=-1;
                x[c][y][x1] = v;
            }
}

// Fully Connected
void fc(float in[C5_OUT][H5][W5], float out[NUM_CLASSES])
{
    float flat[C5_OUT*H5*W5];
    int idx=0;

    for(int c=0;c<C5_OUT;c++)
        for(int y=0;y<H5;y++)
            for(int x=0;x<W5;x++)
                flat[idx++] = in[c][y][x];

    for(int o=0;o<NUM_CLASSES;o++){
        float acc = FC_B[o];
        for(int i=0;i<C5_OUT*H5*W5;i++)
            acc += flat[i] * FC_W[o][i];
        out[o] = acc;
    }
}

// =========== TOP MODULE ============
extern "C" void BNN_VGG_FORWARD(
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

    // ===== Conv1 =====
    xnor_conv<C1_IN,C1_OUT,H0,W0,H1,W1>(input,F1,W1,SCALE1,BIAS1);
    maxpool<C1_OUT,H1,W1,H1,W1>(F1,F1);
    hardtanh<C1_OUT,H1,W1>(F1);

    // ===== Conv2 =====
    xnor_conv<C2_IN,C2_OUT,H1,W1,H2,W2>(F1,F2,W2,SCALE2,BIAS2);
    hardtanh<C2_OUT,H2,W2>(F2);

    // ===== Conv3 =====
    xnor_conv<C3_IN,C3_OUT,H2,W2,H3,W3>(F2,F3,W3,SCALE3,BIAS3);
    maxpool<C3_OUT,H3,W3,H3,W3>(F3,F3);
    hardtanh<C3_OUT,H3,W3>(F3);

    // ===== Conv4 =====
    xnor_conv<C4_IN,C4_OUT,H3,W3,H4,W4>(F3,F4,W4,SCALE4,BIAS4);
    hardtanh<C4_OUT,H4,W4>(F4);

    // ===== Conv5 =====
    xnor_conv<C5_IN,C5_OUT,H4,W4,H5,W5>(F4,F5,W5,SCALE5,BIAS5);
    maxpool<C5_OUT,H5,W5,H5,W5>(F5,F5);
    hardtanh<C5_OUT,H5,W5>(F5);

    // ===== FC =====
    fc(F5,output);
}
