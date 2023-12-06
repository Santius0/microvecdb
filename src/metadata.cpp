#include "metadata.hpp"
#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <utility>

namespace mvdb {

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
            collections.emplace_back(std::move(collection_metadata));
        }
        // std::cout << "LOADED:\n" << *this << std::endl;
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

    void MetadataManager::addCollection(const CollectionMetadata& metadata) {
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
