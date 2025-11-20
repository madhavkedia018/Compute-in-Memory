#ifndef _INTERPRET_HPP_
#define _INTERPRET_HPP_

#include <ap_int.h>

template<
    int TMEM, int PE, int API,
    typename T_THRESH, typename T_OUT>
struct ThresholdsActivation {

    T_THRESH m_thresholds[TMEM][PE][API];

    inline T_OUT activate(int pe, ap_int<16> acc) {
        // Simple threshold: acc > 0 -> 1
        return (acc > 0) ? 1 : 0;
    }
};

template<typename T>
struct PassThroughActivation {
    inline T operator()(T x) {
        return x;
    }
};

#endif
