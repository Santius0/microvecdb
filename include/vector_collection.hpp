#ifndef VECTOR_collection_H
#define VECTOR_collection_H

#include "kv_store.hpp"
#include "vector_index.hpp"
#include "vectorizer.hpp"
#include "serializable.h"
#include <memory>
#include <ostream>

namespace mvdb {

    class CollectionMetadata final: public Serializable {
        std::string name;
        std::string collectionFilePath;
        size_t recordCount{};
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        KvStoreMetadata kv_store_metadata_;
        VectorIndexMetadata vector_index_metadata_;
        VectorizerMetadata vectorizer_metadata_;
        friend class MetadataManager;
        friend class VectorCollection;
    protected:
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
    public:
        CollectionMetadata() = default;
        CollectionMetadata(std::string name, std::string collectionFilePath, const size_t& recordCount,
        const KvStoreMetadata& kv_store_metadata, const VectorIndexMetadata& vector_index_metadata,
        const VectorizerMetadata& vectorizer_metadata);
        ~CollectionMetadata() override = default;
        friend std::ostream& operator<<(std::ostream& os, const CollectionMetadata& obj) {
            return os
                   << "name: " << obj.name
                   << " recordCount: " << obj.recordCount
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
        explicit VectorCollection(const CollectionMetadata& metadata);
        ~VectorCollection() = default;
        // bool add(char* data, float* vector = nullptr);
        // bool remove(uint64_t key);
        // uint64_t* search(float* vectors, size_t num_vectors);
    };

}
#endif // VECTOR_collection_H
