//#ifndef MICROVEC_DB_FLAT_INDEX_H
//#define MICROVEC_DB_FLAT_INDEX_H
//
//#include "index.h"
//#include <vector>
//
//namespace mvdb {
//    class FlatIndex final : public Index {
//        std::vector<vector_t> index_ = {};
//        friend std::ostream& operator<<(std::ostream& os, const FlatIndex& obj);
//        friend std::ostream& operator<<(std::ostream& os, const FlatIndex* obj);
//    protected:
//        void serialize(std::ostream &out) const override;
//        void deserialize(std::istream &in) override;
//    public:
//        explicit FlatIndex(const std::string& index_path, const uint64_t& dims_);
//        ~FlatIndex() override = default;
//        FlatIndex(const FlatIndex&) = delete;
//        FlatIndex& operator=(const FlatIndex&) = delete;
//        void init() override;
//        std::vector<uint64_t> add(const size_t& n, const value_t* data) const override;
//        [[nodiscard]] bool remove(const size_t& n, const faiss::IDSelector& ids) const override;
//        void save() const override;
//        void load() override;
//        void close() override;
//        void search(const std::vector<float>& query, int64_t ids[], float distances[], const long& k = 5) const override;
//    };
//}
//
//#endif // MICROVEC_DB_FLAT_INDEX_H
