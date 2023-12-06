#include "metadata.hpp"
#include "json.hpp"

#include <algorithm>
#include <fstream>
#include <filesystem>

namespace mvdb {

    MetadataManager::MetadataManager(const std::string&  metadataFilePath) : metadataFilePath(metadataFilePath) {
        if(std::filesystem::exists(metadataFilePath)) load();
    }

    MetadataManager::~MetadataManager() {
        save();
    }

    void MetadataManager::load() {
        std::ifstream inFile(metadataFilePath);
        if (inFile.is_open()) {
            nlohmann::json j;
            inFile >> j;
            // collections = j.get<std::vector<CollectionMetadata>>();
            inFile.close();
        }
    }

    void MetadataManager::save() {
        std::ofstream outFile(metadataFilePath);
        if (outFile.is_open()) {
            // nlohmann::json j = collections;
            // outFile << j.dump(4); // Pretty print with 4 spaces
            outFile.close();
        }
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
}
