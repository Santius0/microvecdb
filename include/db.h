#ifndef DB_H
#define DB_H

#include "storage.h"
#include "index.h"
#include "vectorizer.h"
#include <vector>
#include <memory>
#include <filesystem>

namespace mvdb {

    class DB : Serializable {
        std::string dbname_;                                // name of database
        std::string dbpath_;                                // location of database
        idx_t dims_ = 0;                                    // number of dimension each vector in this system will have
        IndexType index_type_ = FAISS_FLAT;                 // type of vector index
        VectorizerModelType
        vec_model_ = VectorizerModelType::FASTTEXT;         // vectorizer model to be used to embedding generation
        std::string metadata_path_;                         // location of binary file that holds DB metadata i.e the values in this object
        std::unique_ptr<Storage> storage_;                  // data storage object
        std::unique_ptr<Index> index_;                      // vector storage object
        idx_t* ids_ = nullptr;                          // ptr to most recent keys array returned for cleanup if necessary
        void make_index_(const std::string& index_path);
        void make_storage_(const std::string& data_path);
        friend std::ostream& operator<<(std::ostream& os, const DB& obj);
        friend std::ostream& operator<<(std::ostream& os, const DB* obj);
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        DB() = default;
        explicit DB(const std::string& dbname = "db", const std::string& dbpath = ".", const idx_t& dims = 300,
                          const IndexType& index_type = IndexType::FLAT,
                          VectorizerModelType vec_model = VectorizerModelType::FASTTEXT);
        ~DB() override;
        void save(const std::string& save_path = "");
        void load(const std::string& load_path = "");
        Index* index();
        Storage* storage();
        [[nodiscard]] idx_t* add_vector(const idx_t& nv, value_t* v);       // add nv vectors to the index
        [[nodiscard]] bool add_data(const idx_t& nv, char* data, idx_t* data_sizes, const DataFormat* data_formats) const;      // take nv pieces of data, generate a vector for each, add vectors to the index, store raw data in kv_store
//        [[nodiscard]] bool add_data_with_vector(const size_t& nv, char* data, size_t* data_sizes, const DataFormat* data_formats, void* v, const DataType& v_d_type = FLOAT) const; // take nv pieces of data and n corresponding vectors, add vectors to the index, add data to the kv_store
        void search_with_vector(const size_t& nq, value_t* query, const long& k, idx_t* ids, value_t* distances) const; // carry out a search using only nq vectors as input
//        [[nodiscard]] SearchResult* search(const size_t& nq, const char* data, const size_t* data_sizes, const DataFormat* data_formats, const long& k, const bool& ret_data) const;  // carry out a search using only raw data as input
        value_t* get(idx_t& n, idx_t* keys) const;
    };

}

#endif //DB_H
