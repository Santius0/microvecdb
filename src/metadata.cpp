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

    void MetadataManager::serialize(std::ostream& out) const {
        serializeString(out, createdTimestamp);
        serializeString(out, modifiedTimestamp);
        serializeSizeT(out, collections.size());
        for (const auto& collection : collections) collection.serialize(out);
    }

    void MetadataManager::deserialize(std::istream& in) {
        createdTimestamp = deserializeString(in);
        modifiedTimestamp = deserializeString(in);
        const size_t num_collections = deserializeSizeT(in);
        for (int i = 0; i < num_collections; i++) {
            CollectionMetadata collection_metadata;
            collection_metadata.deserialize(in);
            collections.emplace_back(std::move(collection_metadata));
        }
    }

    void MetadataManager::load() {
        std::ifstream file(metadataFilePath, std::ios::binary);
        if (!file) throw std::runtime_error("Error opening file for reading: \"" + metadataFilePath + "\"\n");
        deserialize(file);
        std::cout << "LOADED:\n" << *this << std::endl;
        file.close();
    }

    void MetadataManager::save() {
        updateModifiedTimestamp();
        std::ofstream file(metadataFilePath);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + metadataFilePath + "\"\n");
        serialize(file);
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
