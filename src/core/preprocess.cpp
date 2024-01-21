//#include "preprocess.h"
//#include "quantization.h"
//#include <string>
//#include <cassert>
//#include <typeinfo>
//#include <type_traits>
//
//namespace mvdb {
//
////    template <typename T>
////    bool is_fp(T){
////        if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
////            return true;
////        return false;
////    }
//
//
//    // nv = number of vectors being passed
//    // d = dimensionality of each vector
//    template <typename T>
//    void preprocess_vector(const size_t& nv, const size_t& d, T* v,
//                           const bool& opq, const bool& quantize_scaler,
//                           const size_t& n_pre_f, vector_preprocess_func* pre_funcs,
//                           const size_t& n_pos_f, vector_preprocess_func* pos_funcs) {
//         size_t n = nv * d;
//         size_t i;
//         for(i = 0; i < n_pre_f; i++)
//            pre_funcs[i](nv, d, v);
//
//        if(quantize_scaler) {
//            assert(((std::is_same<T, float>::value || std::is_same<T, double>::value)) && "scaler quantization on non float data type is not allowed");
//            auto* q = new value_t[n];
//            scaler_quantization(n, static_cast<float*>(v), q);
//            delete[] static_cast<float*>(v);
//            v = q;
//        }
//
//        for(i = 0; i < nv; i++){
//            // vector will run from position (i * d) to (i * d) + d
//            if(opq) {
//                // TODO: Implement OPQ preprocessing step here
//            }
//            // TODO: Implement any other pre-processing steps here
//        }
//
//        for(i = 0; i < n_pos_f; i++)
//            pos_funcs[i](nv, d, v);
//    }
//
//    template <typename T>
//    void preprocess_data(const size_t& nv, const size_t& d, char* data, T* v,
//                         const DataFormat& d_format,
//                         bool gen_embedding,
//                         const bool& opq, const bool& quantize_scaler,
//                         const size_t& n_pre_f, data_preprocess_func* pre_funcs,
//                         const size_t& n_pos_f, data_preprocess_func* pos_funcs) {
//
////         size_t i;
////         for(i = 0; i < n_pre_f; i++)
////            pre_funcs[i](nv, d, data);
////
////         for(i = 0; i < nv; i++){
////            // vector will run from position (i * d) to (i * d) + d
////         if(d_format == "text") {
////             // TODO: Implement text specific pre-processing
////         } else if(d_format == "image") {
////             // TODO: Implement images specific pre-processing
////         } else if(d_format == "audio") {
////             // TODO: Implement audio specific pre-processing
////         } else if(d_format == "video") {
////             // TODO: Implement video specific pre-processing
////         }
//
////         if(gen_embedding){
////              // TODO: implement embedding generation here
////         }
////
////         // TODO: Implement any general pre-processing steps for all data types
////         }
////         for(i = 0; i < n_pos_f; i++)
////            pos_funcs[i](nv, d, data);
//    }
//
//    template <typename T>
//    void preprocess_data_with_vector(const size_t& nv, const size_t& d, char* data, T* v,
//                                     const DataFormat& d_format,
//                                     const bool& opq, const bool& quantize_scaler,
//                                     const size_t& n_pre_f, data_preprocess_func* pre_funcs,
//                                     const size_t& n_pos_f, data_preprocess_func* pos_funcs) {
//         preprocess_vector(nv, d, v, opq, quantize_scaler, n_pre_f, pre_funcs, n_pos_f, pos_funcs);
//         preprocess_data(nv, d, data, v, d_format, false, opq, quantize_scaler, n_pre_f, pre_funcs, n_pos_f, pos_funcs);
//     }
//
//}
//
