#ifndef INDEX_H
#define INDEX_H

#include "constants.h"
#include "serializable.h"
#include <faiss/IndexIDMap.h>
#include <string>
#include <memory>
#include <ostream>

namespace mvdb {
    using value_t = int8_t;
    using vector_t = value_t*;

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
        friend class VectorDB;
    protected:
        std::string index_path_;
        uint64_t dims_{};
        IndexType index_type_;
        bool is_open_ = false;
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        explicit Index(std::string index_path, const uint64_t &dims, const IndexType &index_type);
        ~Index() override = default;
        Index(const Index&) = delete;
        Index& operator=(const Index&) = delete;
        virtual void init() = 0;
        [[nodiscard]] virtual bool add(const size_t& n, const float* data, uint64_t* keys) const = 0;
        [[nodiscard]] virtual bool remove(const size_t& n, const faiss::IDSelector& ids) const = 0;
        virtual void save() const = 0;
        virtual void load() = 0;
        virtual void open();
        virtual void close() = 0;
        [[nodiscard]] virtual bool is_open() const;
        virtual void search(const int& n, float* query, int64_t* ids, float* distances, const long& k) const = 0;
    };

} // namespace mvdb

#endif // INDEX_H
