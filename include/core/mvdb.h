#ifndef MICROVECDB_MVDB_H
#define MICROVECDB_MVDB_H

#include "db.h"
#include "operators.h"

namespace mvdb {

    template <typename T = float>
    class MVDB {
        std::unique_ptr<DB_<T>> db_;
    public:
        MVDB();
        ~MVDB() = default;

        // system commands
        void create();
        void open();
//        void analyse();
//        void optimise();

        // queries
        bool insert(const idx_t &n, const idx_t &d, const T *v = nullptr, const char *bin = nullptr,
                    const operators::InsertOperatorDataType &input_data_type = operators::InsertOperatorDataType::VECTOR,
                    size_t *sizes = nullptr, const std::string *fp = nullptr);
//        void get();
//        void update();
//        void remove();
//
//        void topk(const idx_t& k, const float& c = 100.0f);
//        void range();
//        void embed();

        inline DB_<T> * get_db_() const {
            return db_.get();
        }
    };
}

#endif //MICROVECDB_MVDB_H
