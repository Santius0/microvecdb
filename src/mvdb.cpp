#include "mvdb.h"

namespace mvdb {


    void MVDB::insert(const idx_t &n, const idx_t &d, const value_t *v, const char *bin,
                      const operators::InsertOperatorDataType &input_data_type,
                      size_t *sizes, const std::string *fp) {
        operators::insert_(db_.get(), n, d, v, bin, input_data_type, sizes, fp);
        if(db_->status()->ok())
            std::cout << "success";
        else
            std::cout << "failure";
        // log things etc
    }

}