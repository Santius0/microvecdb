#ifndef MICROVECDB_ANNOY_INDEX_H
#define MICROVECDB_ANNOY_INDEX_H

#include "index.h"
#include "annoylib.h"
#include "kissrandom.h"

namespace mvdb::index {

    struct AnnoyIndexNamedArgs final : NamedArgs {
        int n_trees = 10;
        int n_threads = -1;
        int search_k = -1;
        AnnoyIndexNamedArgs() = default;
        ~AnnoyIndexNamedArgs() override = default;
    };

    template <typename T = float>
    class MVDBAnnoyIndex final : public Index<T> {
        #ifdef ANNOYLIB_MULTITHREADED_BUILD
        std::unique_ptr<Annoy::AnnoyIndex<int, T, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>> annoy_index_;
        #else
        std::unique_ptr<Annoy::AnnoyIndex<int, T, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>> annoy_index_;
        #endif
        int n_trees = 10, n_threads = -1;
        friend std::ostream& operator<<(std::ostream& os, const MVDBAnnoyIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const MVDBAnnoyIndex<T>* obj);
    protected:
        void save_(const std::string& path) const override;
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        MVDBAnnoyIndex() = default;
        ~MVDBAnnoyIndex() override = default;
        MVDBAnnoyIndex(const MVDBAnnoyIndex&) = delete;
        MVDBAnnoyIndex& operator=(const MVDBAnnoyIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void build(const idx_t &dims, const std::string& path, const std::string& initial_data_path,
                   const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) override;
        void open(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void topk(const idx_t& nq, T* query, const std::string& query_path,
                  const std::string& result_path, idx_t* ids, T* distances, double& peak_wss_mb,
                  const idx_t& k, const DISTANCE_METRIC& distance_metric, const float& c, const NamedArgs* args) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] bool built() const override;
    };

    extern template class MVDBAnnoyIndex<int8_t>;
    extern template class MVDBAnnoyIndex<int16_t>;
    extern template class MVDBAnnoyIndex<uint8_t>;
    extern template class MVDBAnnoyIndex<float>;
}

#endif //MICROVECDB_ANNOY_INDEX_H
