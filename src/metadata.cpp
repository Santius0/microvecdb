#include "metadata.hpp"
#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <utility>

namespace mvdb {

    Metadata::Metadata(const std::string&  metadataFilePath) : metadataFilePath(metadataFilePath) {
        if(std::filesystem::exists(metadataFilePath)) load();
        else {
            createdTimestamp = getCurrentTimeStamp();
            modifiedTimestamp = createdTimestamp;
        }
    }

    Metadata::~Metadata() {
        save();
    }

    void Metadata::serialize(std::ostream& out) const {
        serialize_string(out, createdTimestamp);
        serialize_string(out, modifiedTimestamp);
        serialize_numeric<size_t>(out, collections_.size());
        for (const auto& collection : collections_) collection.serialize(out);
    }

    void Metadata::deserialize(std::istream& in) {
        createdTimestamp = deserialize_string(in);
        modifiedTimestamp = deserialize_string(in);
        const auto num_collections = deserialize_numeric<size_t>(in);
        for (int i = 0; i < num_collections; i++) {
            VectorCollectionMetadata collection_metadata;
            collection_metadata.deserialize(in);
            collections_.emplace_back(std::move(collection_metadata));
        }
    }

    void Metadata::load() {
        std::ifstream file(metadataFilePath, std::ios::binary);
        if (!file) throw std::runtime_error("Error opening file for reading: \"" + metadataFilePath + "\"\n");
        deserialize(file);
        // std::cout << "LOADED:\n" << *this << std::endl;
        file.close();
    }

    void Metadata::save() {
        updateModifiedTimestamp();
        std::ofstream file(metadataFilePath);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + metadataFilePath + "\"\n");
        serialize(file);
        file.close();
    }

    void Metadata::addCollection(const VectorCollectionMetadata& metadata) {
        collections_.push_back(metadata);
    }

    void Metadata::deleteCollection(const std::string& collectionName) {
        // Additional logic to delete actual collection resources...
        collections_.erase(
        std::remove_if(collections_.begin(), collections_.end(),
        [&collectionName](const VectorCollectionMetadata& m) { return m.name == collectionName; }),
        collections_.end());
    }

    void Metadata::updateCreatedTimestamp() {
        createdTimestamp = getCurrentTimeStamp();
    }

    void Metadata::updateModifiedTimestamp() {
        modifiedTimestamp = getCurrentTimeStamp();
    }

}
