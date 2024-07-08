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
        void create(index::IndexType index_type,
                    const uint64_t &dims,
                    const std::string &path,
                    const T *initial_data = nullptr,
                    const std::string& binary_data = "",
                    const size_t *binary_data_sizes = nullptr,
                    const uint64_t& initial_data_size = 0,
                    const NamedArgs *args = nullptr) const;

        void open(const std::string& path) const;

        bool insert(const idx_t &n,
                    const idx_t &d,
                    const void *v = nullptr,
                    const char *bin = nullptr,
                    size_t *sizes = nullptr,
                    const std::string *fp = nullptr) const;

        void get_obj(const uint64_t& n, const uint64_t* keys, std::string* values) const;

        void remove(const idx_t& n,
                    const uint64_t &dims,
                    const T *v = nullptr) const;

        void knn(const idx_t& nq,
                       T* query,
                       std::string query_path,
                       std::string result_path,
                       idx_t* ids, T* distances,
                       double& peak_wss_mb,
                       const idx_t& k = 5,
                       const index::DISTANCE_METRIC& distance_metric = index::DISTANCE_METRIC::L2_DISTANCE,
                       const float& c = 100.0f,
                       const NamedArgs* args = nullptr) const;

        DB_<T> * get_db_() const;
    };
}

#endif //MICROVECDB_MVDB_H
