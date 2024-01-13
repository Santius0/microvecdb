#include "preprocess.h"
#include "quantization.h"
#include <string>
#include <cassert>

namespace mvdb {

    // nv = number of vectors being passed
    // d = dimensionality of each vector
     void preprocess_vector(const size_t& nv, const size_t& d, void* v, const DataType& v_d_type,
                            const bool& opq, const bool& quantize_scaler,
                            const size_t& n_pre_f, vector_preprocess_func* pre_funcs,
                            const size_t& n_pos_f, vector_preprocess_func* pos_funcs) {
         size_t n = nv * d;
         size_t i;
         for(i = 0; i < n_pre_f; i++)
            pre_funcs[i](nv, d, v);

        if(quantize_scaler) {
            assert((v_d_type == FLOAT) && "scaler quantization on non float data type is not allowed");
            auto* q = new value_t[n];
            scaler_quantization(n, static_cast<float*>(v), q);
            delete[] static_cast<float*>(v);
            v = q;
        }

        for(i = 0; i < nv; i++){
            // vector will run from position (i * d) to (i * d) + d
            if(opq) {
                // TODO: Implement OPQ preprocessing step here
            }
            // TODO: Implement any other pre-processing steps here
        }

        for(i = 0; i < n_pos_f; i++)
            pos_funcs[i](nv, d, v);
    }

    void preprocess_data(const size_t& nv, const size_t& d, void* data, void* v,
                         const DataType& v_d_type, const DataFormat& d_format,
                         bool gen_embedding,
                         const bool& opq, const bool& quantize_scaler,
                         const size_t& n_pre_f, data_preprocess_func* pre_funcs,
                         const size_t& n_pos_f, data_preprocess_func* pos_funcs) {

//         size_t i;
//         for(i = 0; i < n_pre_f; i++)
//            pre_funcs[i](nv, d, data);
//
//         for(i = 0; i < nv; i++){
//            // vector will run from position (i * d) to (i * d) + d
//         if(d_format == "text") {
//             // TODO: Implement text specific pre-processing
//         } else if(d_format == "image") {
//             // TODO: Implement images specific pre-processing
//         } else if(d_format == "audio") {
//             // TODO: Implement audio specific pre-processing
//         } else if(d_format == "video") {
//             // TODO: Implement video specific pre-processing
//         }

//         if(gen_embedding){
//              // TODO: implement embedding generation here
//         }
//
//         // TODO: Implement any general pre-processing steps for all data types
//         }
//         for(i = 0; i < n_pos_f; i++)
//            pos_funcs[i](nv, d, data);
    }

    void preprocess_data_with_vector(const size_t& nv, const size_t& d, void* data, void* v,
                                     const DataType& v_d_type, const DataFormat& d_format,
                                     const bool& opq, const bool& quantize_scaler,
                                     const size_t& n_pre_f, data_preprocess_func* pre_funcs,
                                     const size_t& n_pos_f, data_preprocess_func* pos_funcs) {
         preprocess_vector(nv, d, v, v_d_type, opq, quantize_scaler, n_pre_f, pre_funcs, n_pos_f, pos_funcs);
         preprocess_data(nv, d, data, v, v_d_type, d_format, false, opq, quantize_scaler, n_pre_f, pre_funcs, n_pos_f, pos_funcs);
     }

}

