#ifndef INDEX_H
#define INDEX_H

#include "constants.h"
#include "serializable.h"
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

    // Note: This class did not have to be an abstract class with all functions being pure virtual as
    //       many of these functions will have the same functionality across the board for all index types,
    //       however, in an effort to keep things simple, these function will be reimplemented across the board
    //       and this class will remain a fully abstract class. Functioning as an interface.
    class Index : public Serializable {
        friend class DB;
    protected:
        std::string index_path_;                                           // the path to save to and load from
        bool is_open_ = false;
        idx_t dims_ = 0;
        idx_t ntotal_ = 0;
        void serialize(std::ostream &out) const override = 0;
        void deserialize(std::istream &in) override = 0;
    public:
        explicit Index(std::string index_path, const uint64_t &dims) :
            index_path_(std::move(index_path)), dims_(dims) {}
        ~Index() override = default;
        Index(const Index&) = delete;
        Index& operator=(const Index&) = delete;
        [[nodiscard]] virtual IndexType type() const = 0;
        virtual void save() const = 0;          // save current index state and data to location specified via index_path_
        virtual void load() = 0;                // save current index state and data to location specified via index_path_
        virtual void open() = 0;                                            // creates internal index instance or loads one from file if exists. sets is_open to true
        virtual void close(bool force = false) = 0;                                           // shuts off and neatly cleans up internal index object. sets is_open to false
        virtual bool add(const idx_t& n, value_t* data, idx_t* ids) = 0;    // add n to elements to index
        virtual bool remove(const idx_t& n, const idx_t* ids) = 0;          // remove n elements from index
        virtual void search(const idx_t& nq, value_t* query,                // perform searches for nq queries in parallel
                            idx_t* ids, value_t* distances, const idx_t& k) const = 0;
        virtual value_t* get(idx_t& n, idx_t* ids) const = 0;               // reconstruct and return the n vectors specified by ids
        [[nodiscard]] virtual bool is_open() const  = 0;                                  // returns current is_open value
        [[nodiscard]] virtual idx_t dims() const = 0;                                     // returns dimensions of index
        [[nodiscard]] virtual idx_t ntotal() const  = 0;                                  // returns current number of values in index
    };

} // namespace mvdb

#endif // INDEX_H
