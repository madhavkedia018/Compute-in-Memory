#ifndef _MVAU_HPP_
#define _MVAU_HPP_

#include <ap_int.h>

template<
    int MW, int MH,
    int SIMD, int PE>
struct BinaryWeights {
    ap_uint<SIMD> m_weights[PE][MW];
};

#endif
