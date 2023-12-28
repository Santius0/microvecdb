#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#include "constants.hpp"
#include "serializable.hpp"
#include <faiss/Index.h>
#include <faiss/IndexIDMap.h>
#include <string>
#include <memory>
#include <ostream>

namespace mvdb {

    enum class VectorIndexType {
        FLAT = 0,
        IVF = 1
    };

    inline std::ostream& operator<<(std::ostream& os, const VectorIndexType& obj) {
        switch (obj) {
            case VectorIndexType::FLAT:     return os << "FLAT";
            case VectorIndexType::IVF:      return os << "IVF";
            default:                        return os << "invalid index type";
        }
    }

    inline std::ostream& operator<<(std::ostream& os, const VectorIndexType* obj) {
        return os << "*(" << *obj << ")";
    }

    class SearchResult {
        friend std::ostream& operator<<(std::ostream& os, const SearchResult& obj) {
            for(int i = 0; i < obj.size_; i++)
                os << "id: " << obj.ids_[i]
                   << "\tdistance: "
                   << obj.distances_[i]
                   << "\tdata: " << obj.data_[i]
                   << std::endl
                ;
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const SearchResult* obj) {
            return os << "*(" << *obj << ")";
        }
    public:
        int64_t *ids_;
        float* distances_;
        std::string *data_;
        const long size_;
        SearchResult(int64_t* ids, float* distances, std::string* data, const long& size):
        ids_(ids), distances_(distances), data_(data), size_(size) {}
        ~SearchResult() {
            delete[] ids_;
            delete[] distances_;
            delete[] data_;
        };
    };

    class VectorIndex final : public Serializable {
        std::unique_ptr<faiss::Index> faissIndex; // The actual FAISS index
        // std::unique_ptr<faiss::IndexIDMap> faissIndexIDMap;
        // bool id_map = false;
        std::string indexFilePath;
        uint64_t indexDimensions{};
        VectorIndexType indexType;
        friend class VectorDB;
        friend std::ostream& operator<<(std::ostream& os, const VectorIndex& obj);
        friend std::ostream& operator<<(std::ostream& os, const VectorIndex* obj);
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        // Constructor
        explicit VectorIndex(const std::string indexFilePath, const uint64_t& indexDimensions, const VectorIndexType& indexType);

        // Destructor
        ~VectorIndex() override;

        // Non-copyable and non-movable
        VectorIndex(const VectorIndex&) = delete;
        VectorIndex& operator=(const VectorIndex&) = delete;

        // Add data to the index
        std::vector<uint64_t> add(const size_t& n, const float* data, const int64_t* ids = nullptr) const;

        // Remove data from the index
        bool remove(const size_t& n, const faiss::IDSelector& ids) const;

        void save() const;

        void load();

        void search(const std::vector<float>& query, int64_t ids[], float distances[], const long& k = 5) const;

    };

} // namespace mvdb

#endif // VECTOR_INDEX_H
