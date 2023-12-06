#ifndef MICRVECDB_H
#define MICRVECDB_H

#include <vector>
#include <memory>
#include <filesystem>

#include "metadata.hpp"
#include "vector_collection.hpp"

namespace mvdb {

    class MicroVecDB {
        std::string path; // location of database
        std::unique_ptr<MetadataManager> metadata_manager_;
        std::vector<std::unique_ptr<VectorCollection>> collections_;
        std::vector<std::filesystem::path> collection_paths_;
    public:
        MicroVecDB(const std::string& path);
        ~MicroVecDB() = default;
        void create_collection(const std::string& name, const uint64_t& dimensions);
        // std::vector<VectorCollection*> collections();
    };

}

#endif //MICRVECDB_H
