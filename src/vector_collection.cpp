#include "vector_collection.hpp"
#include "utils.hpp"
#include <filesystem>
#include <utility>

namespace mvdb {

    CollectionMetadata::CollectionMetadata(std::string name, std::string collectionFilePath, std::string indexFilePath,
        std::string dataDirectoryPath, std::string model, const size_t& recordCount, const size_t& indexDimensions):
    name(std::move(name)), collectionFilePath(std::move(collectionFilePath)),
    indexFilePath(std::move(indexFilePath)), dataDirectoryPath(std::move(dataDirectoryPath)),
    model(std::move(model)), recordCount(recordCount), indexDimensions(indexDimensions) {
        createdTimestamp = getCurrentTimeStamp();
        modifiedTimestamp = createdTimestamp;
    }

    void CollectionMetadata::serialize(std::ostream& out) const {
        serializeString(out, name);
        serializeString(out, collectionFilePath);
        serializeString(out, indexFilePath);
        serializeString(out, dataDirectoryPath);
        serializeString(out, model);
        serializeNumeric(out, recordCount);
        serializeNumeric(out, indexDimensions);
        serializeString(out, createdTimestamp);
        serializeString(out, modifiedTimestamp);
    }

    void CollectionMetadata::deserialize(std::istream& in) {
        name = deserializeString(in);
        collectionFilePath = deserializeString(in);
        indexFilePath = deserializeString(in);
        dataDirectoryPath = deserializeString(in);
        model = deserializeString(in);
        recordCount = deserializeNumeric(in);
        indexDimensions = deserializeNumeric(in);
        createdTimestamp = deserializeString(in);
        modifiedTimestamp = deserializeString(in);
    }

    VectorCollection::VectorCollection(const CollectionMetadata& metadata) {
        if(!std::filesystem::exists(metadata.collectionFilePath)) std::filesystem::create_directory(metadata.collectionFilePath);
        // vectorizer_ = std::make_unique<Vectorizer>(metadata.model, metadata.indexDimensions);
        // kv_store_ = std::make_unique<KvStore>(metadata.dataDirectoryPath, true, false);
        // vector_index_ = std::make_unique<VectorIndex>(metadata.indexFilePath, metadata.indexFilePath, VectorIndexType::FLAT, dimensions);
    }

}