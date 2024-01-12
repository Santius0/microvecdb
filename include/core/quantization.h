#ifndef MICROVECDB_QUANTIZATION_H
#define MICROVECDB_QUANTIZATION_H

#include "constants.h"
#include <string>
#include <cmath>

namespace mvdb {

    // n = number of values in vector array v
    // v = array containing all vector values, contains nv = n/d vectors, where d = vector dimensionality
    void scaler_quantization(const size_t& n, float* v, value_t* q);
}

#endif