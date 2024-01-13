#ifndef DB_H
#define DB_H

#pragma once

#include "storage.h"
#include "index.h"
#include "vectorizer.h"
#include <vector>
#include <memory>
#include <filesystem>

namespace mvdb {

    class DB : Serializable {
        std::string path_;                          // location of database
        std::string dbname_;                        // name of database
        uint64_t dims_;                             // number of dimension each vector in this system will have
        IndexType index_type_;                      // type of vector index. options are flat, a modified hnsw and the graph based in memory one
        VectorizerModelType vec_model_;             // vectorizer model to be used to embedding generation

        std::string metadata_path_;                 // location of binary file that holds DB metadata i.e the values in this object
        std::unique_ptr<Storage> storage_;         // data storage object
        std::unique_ptr<Index> index_; // vector storage object
        void make_index_(const std::string& index_path);

        friend std::ostream& operator<<(std::ostream& os, const DB& obj);
        friend std::ostream& operator<<(std::ostream& os, const DB* obj);
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        DB() = default;
        explicit DB(const std::string& path, const std::string& dbname = "db", const uint64_t& dims = 300,
                          const IndexType& index_type = IndexType::FAISS_FLAT,
                          VectorizerModelType vec_model = VectorizerModelType::FASTTEXT);
        ~DB() override = default;
        void save(const std::string& save_path = "");
        void load(const std::string& load_path = "");
        Index* index();
        Storage* storage();
        [[nodiscard]] uint64_t* add_vector(const size_t& nv, void* v, const std::string& v_d_type = "float") const;       // add nv vectors to the index
        [[nodiscard]] bool add_data(const size_t& nv, void* data) const;      // take nv pieces of data, generate a vector for each, add vectors to the index, store raw data in kv_store
        [[nodiscard]] bool add_vector_data(const size_t& nv, void* data, void* v) const; // take nv pieces of data and n corresponding vectors, add vectors to the index, add data to the kv_store
        [[nodiscard]] SearchResult* search_with_vector(const size_t& nq, void* query, const long& k, const bool& ret_data) const; // carry out a search using only nq vectors as input
        [[nodiscard]] SearchResult search(void* data, const long& k, const bool& ret_data) const;  // carry out a search using only raw data as input
    };

}

#endif //DB_H
