#ifndef VECTORDB_H
#define VECTORDB_H

#include "metadata.hpp"
#include "vector_collection.hpp"
#include <vector>
#include <memory>
#include <filesystem>

namespace mvdb {

    class VectorDB {
        std::string path; // location of database
        std::string dbname; // location of database
        std::unique_ptr<Metadata> metadata_;
        std::vector<std::shared_ptr<VectorCollection>> collections_;
    public:
        explicit VectorDB(const std::string& path, const std::string& dbname = "db");
        ~VectorDB() = default;
        void create_collection(const std::string& name, const uint64_t& dimensions, const std::string& model);
        VectorCollection* collection(const std::string& name) const;
    };

}

#endif //VECTORDB_H
