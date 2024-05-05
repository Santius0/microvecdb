#include "mvdb.h"

namespace mvdb {

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
}