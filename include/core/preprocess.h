#ifndef MICROVECDB_PREPROCESS_H
#define MICROVECDB_PREPROCESS_H

#include "constants.h"
#include <string>

namespace mvdb {

    typedef void (*vector_preprocess_func)(const size_t& s, const size_t& d, void* v);
    typedef void (*data_preprocess_func)(const size_t& nv, const size_t& d, void* data);

    // nv = number of vectors being passed
    // d = dimensionality of each vector
    template <typename T>
     void preprocess_vector(const size_t& nv, const size_t& d, T* v,
                            const bool& opq = false, const bool& quantize_scaler = false,
                            const size_t& n_pre_f = 0, vector_preprocess_func* pre_funcs = nullptr,
                            const size_t& n_pos_f = 0, vector_preprocess_func* pos_funcs = nullptr);

    template <typename T>
    void preprocess_data(const size_t& nv, const size_t& d, char* data, T* v,
                         const DataFormat& d_format = RAW_TEXT,
                         bool gen_embedding = true,
                         const bool& opq = false, const bool& quantize_scaler = false,
                         const size_t& n_pre_f = 0, data_preprocess_func* pre_funcs = nullptr,
                         const size_t& n_pos_f = 0, data_preprocess_func* pos_funcs = nullptr);

    template <typename T>
    void preprocess_data_with_vector(const size_t& nv, const size_t& d, char* data, T* v,
                                     const DataFormat& d_format = RAW_TEXT,
                                     const bool& opq = false, const bool& quantize_scaler = false,
                                     const size_t& n_pre_f = 0, data_preprocess_func* pre_funcs = nullptr,
                                     const size_t& n_pos_f = 0, data_preprocess_func* pos_funcs = nullptr);

}

#endif // MICROVECDB_PREPROCESS_H