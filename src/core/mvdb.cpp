#include "mvdb.h"
#include "utils.h"

namespace mvdb {

    template <typename T>
    MVDB<T>::MVDB() {
        db_ = std::make_unique<DB_<T>>();
    }

    template<typename T>
    void MVDB<T>::create(index::IndexType index_type,
                         const uint64_t &dims,
                         const std::string &path,
                         const std::string &initial_data_path,
                         const T *initial_data,
                         const std::string& binary_data,
                         const size_t *binary_data_sizes,
                         const uint64_t& initial_data_size,
                         const NamedArgs *args) {
        db_->create(index_type, dims, const_cast<std::string&>(path), initial_data_path, initial_data, binary_data, binary_data_sizes, initial_data_size, args);
    }

    template<typename T>
    void MVDB<T>::open(const std::string& path) {
        db_->open(const_cast<std::string&>(path));
    }

    template <typename T>
    bool MVDB<T>::insert(const idx_t &n, const idx_t &d, const void *v, const char *bin,
                      const operators::InsertOperatorDataType &input_data_type,
                      size_t *sizes, const std::string *fp) {
//        if(input_data_type == operators::InsertOperatorDataType::VECTOR) {
//            operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
//            return db_->status()->success();
//        }
//        else if(input_data_type == operators::InsertOperatorDataType::BINARY) {
//            operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
//            if(db_->status()->ok())
//                operators::insert_(db_.get(), n, d, v, bin, operators::InsertOperatorDataType::VECTOR, sizes, fp);
//            else
//                return db_->status()->success();
//        }
//        else if(input_data_type == operators::InsertOperatorDataType::FILE) {
//            operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
//            if(db_->status()->success())
//                operators::insert_(db_.get(), n, d, v, bin, operators::InsertOperatorDataType::BINARY, sizes, fp);
//            else
//                return db_->status()->success();
//            if(db_->status()->success())
//                operators::insert_(db_.get(), n, d, v, bin, operators::InsertOperatorDataType::VECTOR, sizes, fp);
//            else
//                return db_->status()->success();
//        }
        return db_->status()->success() && db_->status()->ok();
    }

    template <typename T>
    void MVDB<T>::get_obj(const uint64_t& n, const uint64_t* keys, std::string* values) {
        db_->get_obj(n, keys, values);
    }

    template <typename T>
    void MVDB<T>::topk(const idx_t& nq, T* query, std::string query_path, std::string result_path,
                       idx_t* ids, T* distances, double& peak_wss_mb, const idx_t& k,
                       const index::DISTANCE_METRIC& distance_metric, const float& c, const NamedArgs* args) {
        remove_trailing_slashes(query_path);
        remove_trailing_slashes(result_path);
        db_->index()->topk(nq, query, query_path, result_path, ids, distances, peak_wss_mb, k, distance_metric, c, args);
    }

    template class MVDB<int8_t>;
    template class MVDB<int16_t>;
    template class MVDB<uint8_t>;
    template class MVDB<float>;
}