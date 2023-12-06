#include <filesystem>
#include "micrvecdb.hpp"
#include "utils.hpp"
#include "constants.hpp"

#include <iostream>

namespace mvdb {

    MicroVecDB::MicroVecDB(const std::string& path): path(trim(path)) {
        if(!std::filesystem::exists(path))  // if directory doesn't exist create it
            std::filesystem::create_directory(path);
        else if(!std::filesystem::is_directory(path)) // else if file exists but is not a directory throw error
            throw::std::runtime_error("invalid database path \"" + path + "\"");

        metadata_manager_ = std::make_unique<MetadataManager>(this->path + std::filesystem::path::preferred_separator + META_FILE_NAME);

        //don't need this anymore. makes more sense to just load the metadata file and only load from the collections
        //when neccessary
        // for (const auto& entry : std::filesystem::directory_iterator(path)) {
            // if (std::filesystem::is_directory(entry.path())) {
                // this->collections_.push_back(std::unique_ptr<VectorCollection>(VectorCollection::load(entry.path())));
                // this->collection_paths_.push_back(entry.path());
            // }
        // }
    }
}