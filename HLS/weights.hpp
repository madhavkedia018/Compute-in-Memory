#ifndef _WEIGHTS_HPP_
#define _WEIGHTS_HPP_

#include <ap_int.h>

static const ap_uint<64> W0_DUMMY = 0xAAAAAAAAAAAAAAAAULL;

BinaryWeights<L0_SIMD, L0_PE, L0_WMEM> weights0 = {
    { {W0_DUMMY} }
};

BinaryWeights<L1_SIMD, L1_PE, L1_WMEM> weights1 = {
    { {W0_DUMMY} }
};

BinaryWeights<L2_SIMD, L2_PE, L2_WMEM> weights2 = {
    { {W0_DUMMY} }
};

BinaryWeights<L3_SIMD, L3_PE, L3_WMEM> weights3 = {
    { {W0_DUMMY} }
};

BinaryWeights<L4_SIMD, L4_PE, L4_WMEM> weights4 = {
    { {W0_DUMMY} }
};

BinaryWeights<L5_SIMD, L5_PE, L5_WMEM> weights5 = {
    { {W0_DUMMY} }
};

BinaryWeights<L6_SIMD, L6_PE, L6_WMEM> weights6 = {
    { {W0_DUMMY} }
};

BinaryWeights<L7_SIMD, L7_PE, L7_WMEM> weights7 = {
    { {W0_DUMMY} }
};

BinaryWeights<L8_SIMD, L8_PE, L8_WMEM> weights8 = {
    { {W0_DUMMY} }
};

#endif
