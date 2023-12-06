#include <filesystem>
#include "micrvecdb.h"
#include "utils.hpp"
#include "constants.hpp"

namespace mvdb {

    MicroVecDB::MicroVecDB(const std::string& path): path(trim(path)), metadata_manager_(this->path + "\\" + META_FILE_NAME) {
        if(!std::filesystem::exists(path))  // if directory doesn't exist create it
            std::filesystem::create_directory(path);
        else if(!std::filesystem::is_directory(path)) // else if file exists but is not a directory throw error
            throw::std::runtime_error("invalid database path \"" + path + "\"");

       // for (const auto& entry : std::filesystem::directory_iterator(path)) {
            // if (std::filesystem::is_directory(entry.path())) {
                // this->collections_.push_back(std::unique_ptr<VectorCollection>(VectorCollection::load(entry.path())));
                // this->collection_paths_.push_back(entry.path());
            // }
        // }
    }
}