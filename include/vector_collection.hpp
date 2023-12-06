#ifndef VECTOR_collection_H
#define VECTOR_collection_H

#include "kv_store.hpp"
#include "vector_index.hpp"
#include "vectorizer.hpp"
#include <memory>
#include <ostream>

namespace mvdb {

    class CollectionMetadata {
        std::string name;
        std::string collectionFilePath;
        std::string indexFilePath;
        std::string dataDirectoryPath;
        std::string model;
        size_t recordCount{};
        size_t indexDimensions{};
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        void serialize(std::ostream& out) const;
        void deserialize(std::istream& in);
        friend class MetadataManager;
        friend class VectorCollection;
    public:
        CollectionMetadata() = default;
        CollectionMetadata(std::string name, std::string collectionFilePath, std::string indexFilePath,
            std::string dataDirectoryPath, std::string model, const size_t& recordCount, const size_t& indexDimensions);
        ~CollectionMetadata() = default;
        friend std::ostream& operator<<(std::ostream& os, const CollectionMetadata& obj) {
            return os
                   << "name: " << obj.name
                   << " indexFilePath: " << obj.indexFilePath
                   << " dataDirectoryPath: " << obj.dataDirectoryPath
                   << " recordCount: " << obj.recordCount
                   << " indexDimensions: " << obj.indexDimensions
                   << " createdTimestamp: " << obj.createdTimestamp
                   << " modifiedTimestamp: " << obj.modifiedTimestamp;
        }
    };

    class VectorCollection {
        std::unique_ptr<KvStore> kv_store_;
        std::unique_ptr<VectorIndex> vector_index_;
        std::unique_ptr<Vectorizer> vectorizer_;
    public:
        VectorCollection() = default;
        VectorCollection(const CollectionMetadata& metadata);
        ~VectorCollection() = default;
        // bool add(char* data, float* vector = nullptr);
        // bool remove(uint64_t key);
        // uint64_t* search(float* vectors, size_t num_vectors);
    };

}
#endif // VECTOR_collection_H
