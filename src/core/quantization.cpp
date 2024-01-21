#include "quantization.h"

namespace mvdb {

    // n = number of values in vector array v
    // v = array containing all vector values, contains nv = n/d vectors, where d = vector dimensionality
//    void scaler_quantization(const size_t& n, float* v, value_t* q) {
//        #pragma omp parallel for simd
//        for(size_t i = 0; i < n; i++)
//            q[i] = v[i] > 127.0f ? 127 : v[i] < -128.0f ? -128 : std::round(v[i]);
//    }

}