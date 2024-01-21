#ifndef INDEX_H
#define INDEX_H

#include "constants.h"
#include "serializable.h"
#include <faiss/IndexIDMap.h>
#include <string>
#include <memory>
#include <ostream>
#include <filesystem>

namespace mvdb {

    enum IndexType {
        FAISS_FLAT = 0,
        FLAT = 1
    };
    inline std::ostream& operator<<(std::ostream& os, const IndexType& obj) {
        switch (obj) {
            case IndexType::FAISS_FLAT:     return os << "FAISS_FLAT";
            case IndexType::FLAT:           return os << "FLAT";
            default:                        return os << "invalid index type";
        }
    }
    inline std::ostream& operator<<(std::ostream& os, const IndexType* obj) {
        return os << "*(" << *obj << ")";
    }

    class SearchResult {
        friend std::ostream& operator<<(std::ostream& os, const SearchResult& obj) {
            for(int i = 0; i < obj.size_; i++)
                os << "id: " << obj.ids_[i]
                   << "\tdistance: "
                   << obj.distances_[i]
                   << "\tdata: " << obj.data_[i]
                   << std::endl
                ;
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const SearchResult* obj) {
            return os << "*(" << *obj << ")";
        }
    public:
        int64_t *ids_;
        float* distances_;
        std::string *data_;
        const long size_;
        SearchResult(int64_t* ids, float* distances, std::string* data, const long& size):
        ids_(ids), distances_(distances), data_(data), size_(size) {}
        ~SearchResult() {
            delete[] ids_;
            delete[] distances_;
            delete[] data_;
        };
    };

    class Index : public Serializable {
        friend class DB;
        virtual void init() = 0;    // initialises new internal instance of index
    protected:
        std::string index_path_;
        IndexType index_type_ = FAISS_FLAT;
        bool is_open_ = false;
        idx_t dims_ = 0;
        idx_t ntotal_ = 0;

        void serialize(std::ostream &out) const override {
            serialize_string(out, index_path_);
            serialize_numeric<uint64_t>(out, dims_);
            serialize_numeric<uint64_t>(out, static_cast<uint64_t>(index_type_));
        }
        void deserialize(std::istream &in) override {
            index_path_ = deserialize_string(in);
            dims_ = deserialize_numeric<uint64_t>(in);
            index_type_ = static_cast<IndexType>(deserialize_numeric<uint64_t>(in));
        }
    public:
        explicit Index(std::string index_path, const uint64_t &dims, const IndexType &index_type = FAISS_FLAT) :
            index_path_(std::move(index_path)), dims_(dims), index_type_(index_type) {}
        ~Index() override = default;
        Index(const Index&) = delete;
        Index& operator=(const Index&) = delete;
        virtual bool add(const idx_t& n, value_t* data, idx_t* ids) = 0;    // add n to elements to index
        virtual bool remove(const idx_t& n, const idx_t* ids) = 0;          // remove n elements from index
        virtual void save() const = 0;                                      // save current index state and data to location specified via index_path_
        virtual void load() = 0;                                            // load index state and data from location specified via index_path
        virtual void search(const idx_t& nq, value_t* query,                // perform searches for nq queries in parallel
                            idx_t* ids, value_t* distances, const idx_t& k) const = 0;
        virtual value_t* get(idx_t& n, idx_t* ids) const = 0;              // reconstruct and return the n vectors specified by ids

        // creates internal index instance or loads one from file if exists. sets is_open to true
        void open() {
            if(!is_open_) {
                if(std::filesystem::exists(index_path_)) {
                    if (std::filesystem::is_directory(index_path_))
                        throw std::runtime_error(index_path_ + " already exists as directory");
                    load();
                }
                else init();
                is_open_ = true;
            }
        }

        // shuts off and neatly cleans up internal index object. sets is_open to false
        virtual void close(){
            is_open_ = false;
        }

        // returns current is_open value
        [[nodiscard]] bool is_open() const {
            return is_open_;
        }

        // returns dimensions of index
        [[nodiscard]] idx_t dims() const {
            return dims_;
        }

        // returns current number of values in index
        [[nodiscard]] idx_t ntotal() const {
            return ntotal_;
        }
    };

} // namespace mvdb

#endif // INDEX_H
