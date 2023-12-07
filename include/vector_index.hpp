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
        friend class MetadataManager;
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
        friend std::ostream& operator<<(std::ostream& os, const VectorIndex& obj) {
            return os
                   << "\tfaissIndex: " << &obj.faissIndex << std::endl
                   << "\tname: " << obj.name << std::endl
                   << "\tdir: " << obj.dir << std::endl
                   << "\tdims: " << obj.dims << std::endl
                   << "\ttype: " << static_cast<int>(obj.type);
        }

        std::unique_ptr<faiss::Index> faissIndex; // The actual FAISS index
        std::string name;                         // Index's collection name
        std::string dir;                          // Index's directory
        uint64_t dims;                            // Number of dimensions
        VectorIndexType type;                     // Type of the vector index

    public:
        // Constructor
        VectorIndex(std::string  name, std::string  dir,
             VectorIndexType type = VectorIndexType::FLAT, uint64_t dims = 300);

        // Create an index from a file. Basically just a wrapper for the constructor
        static VectorIndex* create(const std::string& name, const std::string& dir,
            const VectorIndexType& type, const uint64_t& dims);

        // Load an index from a file
        static VectorIndex* load(const std::string& name, const std::string& dir);

        // Destructor
        ~VectorIndex() = default;

        // Non-copyable and non-movable
        VectorIndex(const VectorIndex&) = delete;
        VectorIndex& operator=(const VectorIndex&) = delete;

        // Add data to the index
        [[nodiscard]] bool add(const size_t& n, const float* data) const;

        // Remove data from the index
        [[nodiscard]] bool remove(const size_t& n, const faiss::IDSelector& ids) const;

        // Save the index to a file
        [[nodiscard]] bool save() const;

    };

} // namespace mvdb

#endif // VECTOR_INDEX_H
