#ifndef MICROVECDB_MVDB_H
#define MICROVECDB_MVDB_H

#include "db.h"
#include "operators.h"
#include "index.h"

namespace mvdb {

    template <typename T = float>
    class MVDB {
        std::unique_ptr<DB_<T>> db_;
    public:
        MVDB();
        ~MVDB() = default;

        // system commands
        void create(index::IndexType index_type, const uint64_t &dims, const std::string &path,
                    const std::string &initial_data_path = "",
                    const T *initial_data = nullptr, const uint64_t& initial_data_size = 0,
                    const NamedArgs *args = nullptr);
        void open(const std::string& path);
//        void analyse();
//        void optimise();

        // queries
        bool insert(const idx_t &n, const idx_t &d, const void *v = nullptr, const char *bin = nullptr,
                    const operators::InsertOperatorDataType &input_data_type = operators::InsertOperatorDataType::VECTOR,
                    size_t *sizes = nullptr, const std::string *fp = nullptr);
//        void get();
//        void update();
//        void remove();
//
        void topk(const idx_t& nq, T* query, std::string query_path, std::string result_path, idx_t* ids, T* distances, const idx_t& k = 5,
                  const index::DISTANCE_METRIC& distance_metric = index::DISTANCE_METRIC::L2_DISTANCE,
                  const float& c = 100.0f);
//        void range();
//        void embed();

        inline DB_<T> * get_db_() const {
            return db_.get();
        }
    };
}

#endif //MICROVECDB_MVDB_H
