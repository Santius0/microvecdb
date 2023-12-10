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

    class VectorIndexMetadata : Serializable {
        std::string indexFilePath;
        uint64_t indexDimensions{};
        VectorIndexType indexType;
        friend class VectorIndex;
        friend class VectorCollectionMetadata;
        friend class Metadata;
    protected:
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
    public:
        VectorIndexMetadata() = default;
        VectorIndexMetadata(std::string indexFilePath, const uint64_t& indexDimensions,
        const VectorIndexType& indexType);
        ~VectorIndexMetadata() override = default;
    };

    class VectorIndex {
        std::unique_ptr<faiss::Index> faissIndex; // The actual FAISS index
        // std::unique_ptr<faiss::IndexIDMap> faissIndexIDMap;
        // bool id_map = false;
        std::string indexFilePath;
        uint64_t indexDimensions{};

    public:
        // Constructor
        explicit VectorIndex(const VectorIndexMetadata& metadata);

        // Destructor
        ~VectorIndex();

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
