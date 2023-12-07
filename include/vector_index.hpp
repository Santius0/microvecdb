#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#include "serializable.h"
#include <faiss/Index.h>
#include <string>
#include <memory>
#include <ostream>

namespace mvdb {

    enum class VectorIndexType {
        FLAT = 0,
        IVF = 1
    };

    class VectorIndexMetadata final: Serializable {
        std::string indexFilePath;
        uint64_t indexDimensions{};
        VectorIndexType indexType;
        friend class VectorIndex;
        friend class CollectionMetadata;
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
        std::string indexFilePath;

    public:
        // Constructor
        explicit VectorIndex(const VectorIndexMetadata& metadata);

        // Destructor
        ~VectorIndex();

        // Non-copyable and non-movable
        VectorIndex(const VectorIndex&) = delete;
        VectorIndex& operator=(const VectorIndex&) = delete;

        // Add data to the index
        [[nodiscard]] bool add(const size_t& n, const float* data) const;

        // Remove data from the index
        [[nodiscard]] bool remove(const size_t& n, const faiss::IDSelector& ids) const;

        void save() const;

        void load();

    };

} // namespace mvdb

#endif // VECTOR_INDEX_H
