#ifndef _BNN_LIBRARY_H_
#define _BNN_LIBRARY_H_

#include <hls_stream.h>
#include <ap_int.h>

using namespace hls;

/*******************************
 *  STREAMING WIDTH CONVERTERS
 *******************************/
template<int InWidth, int OutWidth, int NumInWords>
void StreamingDataWidthConverter_Batch(
    stream<ap_uint<InWidth>> &in,
    stream<ap_uint<OutWidth>> &out,
    const unsigned int reps)
{
#pragma HLS INLINE
    const int in_per_out = OutWidth / InWidth;

    for (unsigned int rep = 0; rep < reps; rep++) {
        for (int i = 0; i < NumInWords / in_per_out; i++) {
#pragma HLS PIPELINE II=1
            ap_uint<OutWidth> temp = 0;
            for (int j = 0; j < in_per_out; j++) {
                ap_uint<InWidth> v = in.read();
                temp.range((j+1)*InWidth-1, j*InWidth) = v;
            }
            out.write(temp);
        }
    }
}

/******************************************
 * MEM2STREAM AND STREAM2MEM
 ******************************************/
template<int WordWidth, int NumWords>
void Mem2Stream_Batch(ap_uint<WordWidth>* mem, stream<ap_uint<WordWidth>>& out, unsigned int reps)
{
#pragma HLS INLINE
    for (unsigned int r = 0; r < reps; r++) {
        for (int i = 0; i < NumWords; i++) {
#pragma HLS PIPELINE II=1
            out.write(mem[i]);
        }
    }
}

template<int WordWidth, int NumWords>
void Stream2Mem_Batch(stream<ap_uint<WordWidth>>& in, ap_uint<WordWidth>* mem, unsigned int reps)
{
#pragma HLS INLINE
    for (unsigned int r = 0; r < reps; r++) {
        for (int i = 0; i < NumWords; i++) {
#pragma HLS PIPELINE II=1
            mem[i] = in.read();
        }
    }
}

/*************************************
 *  MAXPOOL (STRIDE 2)
 *************************************/
template<int FM_DIM, int STRIDE, int CHANNELS>
void StreamingMaxPool_Batch(
    stream<ap_uint<CHANNELS>> &in,
    stream<ap_uint<CHANNELS>> &out,
    unsigned int reps)
{
#pragma HLS INLINE

    for (unsigned int rep = 0; rep < reps; rep++) {
        for (int r = 0; r < FM_DIM; r += STRIDE) {
            for (int c = 0; c < FM_DIM; c += STRIDE) {
#pragma HLS PIPELINE II=1
                ap_uint<CHANNELS> out_word = 0;
                ap_uint<CHANNELS> x0 = in.read();
                ap_uint<CHANNELS> x1 = in.read();
                ap_uint<CHANNELS> x2 = in.read();
                ap_uint<CHANNELS> x3 = in.read();

                for (int ch = 0; ch < CHANNELS; ch++) {
                    bool maxbit = x0[ch] | x1[ch] | x2[ch] | x3[ch];
                    out_word[ch] = maxbit;
                }

                out.write(out_word);
            }
        }
    }
}

/*************************************
 *  XNOR-Based Convolution Layer
 *************************************/
template<
    int K, int IFM_CH, int IFM_DIM,
    int OFM_CH, int OFM_DIM,
    int SIMD, int PE,
    template<typename> class ActFunc,
    template<typename> class AccFunc>
void ConvLayer_Batch(
    hls::stream<ap_uint<IFM_CH>> &in,
    hls::stream<ap_uint<OFM_CH>> &out,
    BinaryWeights<SIMD, PE> &weights,
    ThresholdsActivation<> &thr,
    unsigned int reps)
{
#pragma HLS INLINE

    for (unsigned int rep = 0; rep < reps; rep++) {
        for (int i = 0; i < OFM_DIM * OFM_DIM; i++) {
#pragma HLS PIPELINE II=1
            ap_uint<OFM_CH> out_ch = 0;

            for (int pe = 0; pe < PE; pe++) {
                ap_int<8> acc = 0;
                for (int s = 0; s < SIMD; s++) {
                    bool a = in.read()[s];
                    bool w = weights.m_weights[pe][s];

                    acc += (a == w) ? 1 : -1;
                }
                out_ch[pe] = (acc > 0);
            }
            out.write(out_ch);
        }
    }
}

/********************************************
 *  Fully Connected Layer (XNOR Based)
 ********************************************/
template<int MW, int MH, int SIMD, int PE>
void StreamingFCLayer_Batch(
    stream<ap_uint<SIMD>> &in,
    stream<ap_uint<PE>> &out,
    BinaryWeights<SIMD, PE> &weights,
    ThresholdsActivation<> &thr,
    unsigned int reps,
    ap_resource_attr_t)
{
#pragma HLS INLINE

    for (unsigned int r = 0; r < reps; r++) {
        for (int outch = 0; outch < MH / PE; outch++) {

#pragma HLS PIPELINE II=1
            ap_uint<PE> out_word = 0;

            for (int pe = 0; pe < PE; pe++) {
                ap_int<16> acc = 0;

                for (int s = 0; s < SIMD; s++) {
                    bool a = in.read()[s];
                    bool w = weights.m_weights[outch * PE + pe][s];
                    acc += (a == w) ? 1 : -1;
                }
                out_word[pe] = (acc > 0);
            }

            out.write(out_word);
        }
    }
}

#endif
