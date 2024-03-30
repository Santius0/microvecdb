#ifndef INDEX_H
#define INDEX_H

#include "constants.h"
#include "distances.h"
#include "serializable.h"
#include <string>
#include <memory>
#include <ostream>

namespace mvdb {

    enum IndexType : unsigned char {
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


    // Note: This class did not have to be an abstract class with all functions being pure virtual as
    //       many of these functions will have the same functionality across the board for all index types,
    //       however, in an effort to keep things simple, these function will be reimplemented across the board
    //       and this class will remain a fully abstract class. Functioning as an interface.
    class Index : public Serializable {
        friend class DB;
    protected:
        idx_t dims_ = 0;
        idx_t ntotal_ = 0;
        void serialize(std::ostream &out) const override = 0;
        void deserialize(std::istream &in) override = 0;
    public:
        explicit Index(const idx_t &dims, const std::string& path = "") : dims_(dims) {}
        ~Index() override = default;
        Index(const Index&) = delete;
        Index& operator=(const Index&) = delete;
        [[nodiscard]] virtual IndexType type() const = 0;
        virtual void save(const std::string& path) const = 0;               // save current index state and data to location specified via index_path_
        virtual void load(const std::string& path) = 0;                     // load current index state and data from location specified via index_path_
        virtual bool add(const idx_t& n, value_t* data, idx_t* ids) = 0;    // add n to elements to index
        virtual bool remove(const idx_t& n, const idx_t* ids) = 0;          // remove n elements from index
        virtual void search(const idx_t& nq, value_t* query,                // perform searches for nq queries in parallel
                            idx_t* ids, value_t* distances,
                            const idx_t& k,
                            const DISTANCE_METRIC& distance_metric) const = 0;
        virtual value_t* get(idx_t& n, idx_t* ids) const = 0;               // reconstruct and return the n vectors specified by ids
        virtual value_t* get_all() const = 0;                     // reconstruct and return all vectors
        [[nodiscard]] virtual idx_t dims() const = 0;                       // returns dimensions of index
        [[nodiscard]] virtual idx_t ntotal() const  = 0;                    // returns current number of values in index
    };

} // namespace mvdb

#endif // INDEX_H
