#ifndef MICRVECDB_H
#define MICRVECDB_H

#include "metadata.hpp"
#include "vector_collection.hpp"
#include <vector>
#include <memory>
#include <filesystem>

namespace mvdb {

    class MicroVecDB {
        std::string path; // location of database
        std::string dbname; // location of database
        std::unique_ptr<Metadata> metadata_;
        std::vector<std::shared_ptr<VectorCollection>> collections_;
    public:
        explicit MicroVecDB(const std::string& path, const std::string& dbname = "db");
        ~MicroVecDB() = default;
        void create_collection(const std::string& name, const uint64_t& dimensions, const std::string& model);
        VectorCollection* collection(const std::string& name) const;
    };

}

#endif //MICRVECDB_H
