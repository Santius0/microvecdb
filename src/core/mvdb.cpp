#include "mvdb.h"

namespace mvdb {

    template<typename T>
    void MVDB<T>::create(index::IndexType index_type, const uint64_t &dims, const std::string &path,
                         const std::string &initial_data_path, const T *initial_data, const uint64_t& initial_data_size,
                         const NamedArgs *args) {
        db_ = std::make_unique<DB_<T>>();
        if(args)
            db_->create(index_type, dims, path, initial_data_path, initial_data, initial_data_size, args);
        else
            db_->create(index_type, dims, path, initial_data_path, initial_data, initial_data_size);
    }

    template<typename T>
    void MVDB<T>::open(const std::string& path) {
        db_->open(path);
    }

    template <typename T>
    bool MVDB<T>::insert(const idx_t &n, const idx_t &d, const T *v, const char *bin,
                      const operators::InsertOperatorDataType &input_data_type,
                      size_t *sizes, const std::string *fp) {
        if(input_data_type == operators::InsertOperatorDataType::VECTOR) {
            operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
            return db_->status()->success();
        }
        else if(input_data_type == operators::InsertOperatorDataType::BINARY) {
            operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
            if(db_->status()->ok())
                operators::insert_(db_.get(), n, d, v, bin, operators::InsertOperatorDataType::VECTOR, sizes, fp);
            else
                return db_->status()->success();
        }
        else if(input_data_type == operators::InsertOperatorDataType::FILE) {
            operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
            if(db_->status()->success())
                operators::insert_(db_.get(), n, d, v, bin, operators::InsertOperatorDataType::BINARY, sizes, fp);
            else
                return db_->status()->success();
            if(db_->status()->success())
                operators::insert_(db_.get(), n, d, v, bin, operators::InsertOperatorDataType::VECTOR, sizes, fp);
            else
                return db_->status()->success();
        }
        return db_->status()->success() && db_->status()->ok();
    }

    template <typename T>
    void MVDB<T>::topk(const uint64_t& k, const float& c){

    }

}