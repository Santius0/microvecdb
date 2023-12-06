#include "vector_collection.hpp"
#include "constants.hpp"
#include "utils.hpp"
#include <filesystem>
#include <utility>

namespace mvdb {

    CollectionMetadata::CollectionMetadata(std::string name, std::string indexFilePath, std::string dataDirectoryPath,
            const size_t& recordCount, const size_t& indexDimensions): name(std::move(name)),
    indexFilePath(std::move(indexFilePath)), dataDirectoryPath(std::move(dataDirectoryPath)),
    recordCount(recordCount), indexDimensions(indexDimensions) {
        createdTimestamp = getCurrentTimeStamp();
        modifiedTimestamp = createdTimestamp;
    }

    void CollectionMetadata::serialize(std::ostream& out) const {
        serializeString(out, name);
        serializeString(out, indexFilePath);
        serializeString(out, dataDirectoryPath);
        serializeSizeT(out, recordCount);
        serializeSizeT(out, indexDimensions);
        serializeString(out, createdTimestamp);
        serializeString(out, modifiedTimestamp);
    }

    void CollectionMetadata::deserialize(std::istream& in) {
        name = deserializeString(in);
        indexFilePath = deserializeString(in);
        dataDirectoryPath = deserializeString(in);
        recordCount = deserializeSizeT(in);
        indexDimensions = deserializeSizeT(in);
        createdTimestamp = deserializeString(in);
        modifiedTimestamp = deserializeString(in);
    }

    VectorCollection::VectorCollection(const std::string& model, const uint64_t& dimensions, const std::string& path) {
        if(!std::filesystem::exists(path)) std::filesystem::create_directory(path);
        // vectorizer_ = std::make_unique<Vectorizer>(model, dimensions);
        // kv_store_ = std::make_unique<KvStore>(this->kv_path, false, false);
        // vector_index_ = std::make_unique<VectorIndex>(this->name, this->path, VectorIndexType::FLAT, this->dimensions);
    }

    VectorCollection* VectorCollection::load(const std::string& path) {
        // return new VectorCollection();
        return nullptr;
    }

}