#ifndef MICRVECDB_H
#define MICRVECDB_H

#include <vector>
#include <memory>
#include <filesystem>

#include "metadata.h"
#include "vector_collection.h"

namespace mvdb {

    class MicroVecDB {
        std::vector<std::unique_ptr<VectorCollection>> collections_;
        std::string path; // location of database
        std::vector<std::filesystem::path> collection_paths_;
        MetadataManager metadata_manager_;
    public:
        explicit MicroVecDB(const std::string& path);
        ~MicroVecDB() = default;
        void create_collection(const std::string& name, const uint64_t& dimensions);
        // std::vector<VectorCollection*> collections();
    };

}

#endif //MICRVECDB_H
