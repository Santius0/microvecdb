#include "vector_collection.hpp"
#include "utils.hpp"
#include <filesystem>
#include <utility>

namespace mvdb {

    CollectionMetadata::CollectionMetadata(std::string name, std::string collectionFilePath, const size_t& recordCount,
        const KvStoreMetadata& kv_store_metadata, const VectorIndexMetadata& vector_index_metadata,
        const VectorizerMetadata& vectorizer_metadata):
    name(std::move(name)), collectionFilePath(std::move(collectionFilePath)), recordCount(recordCount) {
        createdTimestamp = getCurrentTimeStamp();
        modifiedTimestamp = createdTimestamp;

        kv_store_metadata_ = kv_store_metadata;
        vector_index_metadata_ = vector_index_metadata;
        vectorizer_metadata_ = vectorizer_metadata;
    }

    void CollectionMetadata::serialize(std::ostream& out) const {
        serializeString(out, name);
        serializeString(out, collectionFilePath);
        serializeUInt64T(out, recordCount);
        serializeString(out, createdTimestamp);
        serializeString(out, modifiedTimestamp);
        kv_store_metadata_.serialize(out);
        vector_index_metadata_.serialize(out);
        vectorizer_metadata_.serialize(out);
    }

    void CollectionMetadata::deserialize(std::istream& in) {
        name = deserializeString(in);
        collectionFilePath = deserializeString(in);
        recordCount = deserializeUInt64T(in);
        createdTimestamp = deserializeString(in);
        modifiedTimestamp = deserializeString(in);
        kv_store_metadata_.deserialize(in);
        vector_index_metadata_.deserialize(in);
        vectorizer_metadata_.deserialize(in);
    }

    VectorCollection::VectorCollection(const CollectionMetadata& metadata) {
        if(!std::filesystem::exists(metadata.collectionFilePath)) std::filesystem::create_directory(metadata.collectionFilePath);
        kv_store_ = std::make_unique<KvStore>(metadata.kv_store_metadata_);
        vector_index_ = std::make_unique<VectorIndex>(metadata.vector_index_metadata_);
        vectorizer_ = std::make_unique<Vectorizer>(metadata.vectorizer_metadata_);
    }

}