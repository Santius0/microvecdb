#include "metadata.hpp"
#include "utils.hpp"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <utility>

namespace mvdb {

    CollectionMetadata::CollectionMetadata(std::string name, std::string indexFilePath, std::string dataDirectoryPath,
            const size_t& recordCount, const size_t& indexSize): name(std::move(name)),
    indexFilePath(std::move(indexFilePath)), dataDirectoryPath(std::move(dataDirectoryPath)),
    recordCount(recordCount), indexSize(indexSize) {
        createdTimestamp = getCurrentTimeStamp();
        modifiedTimestamp = createdTimestamp;
    }

    void CollectionMetadata::serialize(std::ostream& out) const {
        serializeString(out, name);
        serializeString(out, indexFilePath);
        serializeString(out, dataDirectoryPath);
        serializeSizeT(out, recordCount);
        serializeSizeT(out, indexSize);
        serializeString(out, createdTimestamp);
        serializeString(out, modifiedTimestamp);
    }

    void CollectionMetadata::deserialize(std::istream& in) {
        name = deserializeString(in);
        indexFilePath = deserializeString(in);
        dataDirectoryPath = deserializeString(in);
        recordCount = deserializeSizeT(in);
        indexSize = deserializeSizeT(in);
        createdTimestamp = deserializeString(in);
        modifiedTimestamp = deserializeString(in);
    }

    MetadataManager::MetadataManager(const std::string&  metadataFilePath) : metadataFilePath(metadataFilePath) {
        if(std::filesystem::exists(metadataFilePath)) load();
        else {
            createdTimestamp = getCurrentTimeStamp();
            modifiedTimestamp = createdTimestamp;
        }
    }

    MetadataManager::~MetadataManager() {
        save();
    }

    void MetadataManager::load() {
        std::ifstream file(metadataFilePath, std::ios::binary);
        if (!file) throw std::runtime_error("Error opening file for reading: \"" + metadataFilePath + "\"\n");
        createdTimestamp = deserializeString(file);
        modifiedTimestamp = deserializeString(file);
        size_t num_collections = deserializeSizeT(file);
        for (int i = 0; i < num_collections; i++) {
            CollectionMetadata collection_metadata;
            collection_metadata.deserialize(file);
            collections.emplace_back(collection_metadata);
        }
        file.close();
    }

    void MetadataManager::save() {
        updateModifiedTimestamp();
        std::ofstream file(metadataFilePath);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + metadataFilePath + "\"\n");
        serializeString(file, createdTimestamp);
        serializeString(file, modifiedTimestamp);
        serializeSizeT(file, collections.size());
        for (const auto& collection : collections) collection.serialize(file);
        file.close();
    }

    void MetadataManager::createCollection(const CollectionMetadata& metadata) {
        collections.push_back(metadata);
    }

    void MetadataManager::deleteCollection(const std::string& collectionName) {
        // Additional logic to delete actual collection resources...
        collections.erase(
        std::remove_if(collections.begin(), collections.end(),
        [&collectionName](const CollectionMetadata& m) { return m.name == collectionName; }),
        collections.end());
    }

    void MetadataManager::updateCreatedTimestamp() {
        createdTimestamp = getCurrentTimeStamp();
    }

    void MetadataManager::updateModifiedTimestamp() {
        modifiedTimestamp = getCurrentTimeStamp();
    }

}
