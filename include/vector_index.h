#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#include <faiss/Index.h>
#include <string>
#include <memory>
#include <ostream>

namespace mvdb {

    enum class VectorIndexType {
        FLAT = 0,
        IVF = 1
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
        VectorIndex(const std::string& name, const std::string& dir,
             VectorIndexType type = VectorIndexType::FLAT, uint64_t dims = 300);

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

        // Load the index from a file
        static VectorIndex* load(const std::string& name, const std::string& dir);
    };

} // namespace mvdb

#endif // VECTOR_INDEX_H
