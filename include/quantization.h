#ifndef MICROVECDB_QUANTIZATION_H
#define MICROVECDB_QUANTIZATION_H

#include "index/index.h"
#include <cmath>

namespace mvdb {

    inline void scaler_quantization(const size_t& n, float* v, value_t* q){
        if(q == nullptr) q = new value_t[n];
        #pragma omp parallel for simd
        for(size_t i = 0; i < n; i++){
            q[i] = round(v[i]);
        }
    }

}

#endif