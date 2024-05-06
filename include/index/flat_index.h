//#ifndef MICROVECDB_FLAT_INDEX_H
//#define MICROVECDB_FLAT_INDEX_H
//
//#include "index.h"
//
//namespace mvdb::index {
//
//    template <typename T = float>
//    class FlatIndex final : public Index<T> {
//        std::shared_ptr<std::vector<T>> index_;    // The actual index
//        friend std::ostream& operator<<(std::ostream& os, const FlatIndex<T>& obj);
//        friend std::ostream& operator<<(std::ostream& os, const FlatIndex<T>* obj);
//    protected:
//        void save_(const std::string& path) const override;
//    public:
//        void serialize_(std::ostream &out) const override;
//        void deserialize_(std::istream &in) override;
//        FlatIndex() = default;
//        ~FlatIndex() override = default;
//        FlatIndex(const FlatIndex&) = delete;
//        FlatIndex& operator=(const FlatIndex&) = delete;
//        [[nodiscard]] index::IndexType type() const override;
//        void build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) override;
//        void open(const std::string& path) override;
//        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
//        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
//        void search(const idx_t& nq, T* query, idx_t* ids, T* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
//        T* get(idx_t& n, idx_t* keys) const override;
//        T* get_all() const override;
//        [[nodiscard]] idx_t dims() const override;
//        [[nodiscard]] idx_t ntotal() const override;
//        [[nodiscard]] T* index() const;
//    };
//
//    extern template class FlatIndex<int8_t>;
//    extern template class FlatIndex<int16_t>;
//    extern template class FlatIndex<int32_t>;
//    extern template class FlatIndex<int64_t>;
//    extern template class FlatIndex<uint8_t>;
//    extern template class FlatIndex<uint16_t>;
//    extern template class FlatIndex<uint32_t>;
//    extern template class FlatIndex<uint64_t>;
//    extern template class FlatIndex<float>;
//    extern template class FlatIndex<double>;
//
//} // namespace mvdb
//
//
//#endif //MICROVECDB_FLAT_INDEX_H
