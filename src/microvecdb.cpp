#include "micrvecdb.hpp"
#include "utils.hpp"
#include "constants.hpp"
#include <filesystem>

namespace mvdb {

    MicroVecDB::MicroVecDB(const std::string& path, const std::string& dbname): path(trim(path)), dbname(dbname) {
        if(!std::filesystem::exists(path))  // if directory doesn't exist create it
            std::filesystem::create_directory(path);
        else if(!std::filesystem::is_directory(path)) // else if file exists but is not a directory throw error
            throw::std::runtime_error("invalid database path \"" + path + "\"");

        metadata_manager_ = std::make_unique<MetadataManager>(this->path
            + std::filesystem::path::preferred_separator + dbname + META_FILE_EXTENSION);

        //don't need this anymore. makes more sense to just load the metadata file and only load from the collections
        //when neccessary
        // for (const auto& entry : std::filesystem::directory_iterator(path)) {
            // if (std::filesystem::is_directory(entry.path())) {
                // this->collections_.push_back(std::unique_ptr<VectorCollection>(VectorCollection::load(entry.path())));
                // this->collection_paths_.push_back(entry.path());
            // }
        // }
    }

    void MicroVecDB::create_collection(const std::string& name, const uint64_t& dimensions, const std::string& model) {
        const std::string collectionFilePath = path + std::filesystem::path::preferred_separator + name;
        const std::string indexFilePath = collectionFilePath + std::filesystem::path::preferred_separator + name + INDEX_EXT;
        const std::string dataDirectoryPath = collectionFilePath + std::filesystem::path::preferred_separator + name + KV_STORE_EXT;

        const auto collection_metadata = CollectionMetadata(name, indexFilePath, dataDirectoryPath, 0, dimensions);
        metadata_manager_->addCollection(collection_metadata);

        auto* vc = new VectorCollection(model, dimensions, collectionFilePath);
        collections_.emplace_back(vc);

        metadata_manager_->save();
    }

}