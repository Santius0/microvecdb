#ifndef VECTORDB_H
#define VECTORDB_H

#pragma once

#include "kv_store.hpp"
#include "schema.h"
#include "vector_index.hpp"
#include "vectorizer.hpp"

#include "metadata.hpp"
#include "vector_collection.hpp"
#include <vector>
#include <memory>
#include <filesystem>

namespace mvdb {

    class VectorDB : Serializable {
        std::string path_;                          // location of database
        std::string dbname_;                        // name of database
        uint64_t dims_;                             // number of dimension each vector in this system will have
        VectorIndexType index_type_;                      // type of vector index. options are flat, a modified hnsw and the graph based in memory one
        VectorizerModelType vec_model_;             // vectorizer model to be used to embedding generation

        std::string metadata_path_;                 // location of binary file that holds DB metadata i.e the values in this object
        std::unique_ptr<KvStore> kv_store_;         // data storage object
        std::unique_ptr<VectorIndex> vector_index_; // vector storage object

        friend std::ostream& operator<<(std::ostream& os, const VectorDB& obj);
        friend std::ostream& operator<<(std::ostream& os, const VectorDB* obj);
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        VectorDB() = default;
        explicit VectorDB(const std::string& path, const std::string& dbname = "db", const uint64_t& dims = 300,
                          const VectorIndexType& index_type = VectorIndexType::FLAT,
                          VectorizerModelType vec_model = VectorizerModelType::FASTTEXT);
        ~VectorDB() override = default;
        void save(const std::string& save_path = "");
        void load(const std::string& load_path = "");
        VectorIndex* index();
        KvStore* storage();
        bool add_data_vector(const std::string& data, float* vec) const;
        bool add_data(const std::string& data) const;
        [[nodiscard]] SearchResult search_with_vector(const std::vector<float>& query, const long& k, const bool& ret_data) const;
        [[nodiscard]] SearchResult search(const std::string& data, const long& k, const bool& ret_data) const;
    };

}

#endif //VECTORDB_H
