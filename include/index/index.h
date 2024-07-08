#ifndef INDEX_H
#define INDEX_H

#include "constants.h"
#include "distances.h"
#include "serializable.h"
#include "args.h"
#include <string>
#include <memory>
#include <ostream>

// build - builds actual in file object
// open - open file path
// topk - return top-k elements
// add - add n vectors
// remove - remove n vectors

// is a save method a neccessary? => probably not right? => save is only called when data is added or removed and this
// should automatically be reflected in the end file or we could have a save method and just have it be private

namespace mvdb::index {

    enum IndexType : unsigned char {
        FAISS_FLAT = 0,
        ANNOY = 1,
        SPANN = 2
    };

    inline std::ostream& operator<<(std::ostream& os, const IndexType& obj) {
        switch (obj) {
            case IndexType::FAISS_FLAT:     return os << "FAISS_FLAT";
            case IndexType::ANNOY:          return os << "ANNOY";
            case IndexType::SPANN:          return os << "SPANN";
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
    template <typename T>
    class Index : public Serializable {
//        friend class DB_;
    protected:
        idx_t dims_ = 0;
        idx_t ntotal_ = 0;
        bool built_ = false;
        virtual void save_(const std::string& path) const = 0;               // save current index state and data to location specified via index_path_
    public:
        void serialize_(std::ostream &out) const override = 0;
        void deserialize_(std::istream &in) override = 0;
        Index() = default;
        ~Index()  override = default;
        Index(const Index&) = delete;
        Index& operator=(const Index&) = delete;
        [[nodiscard]] virtual IndexType type() const = 0;
        virtual void build(const idx_t &dims,
                           const std::string& path,
                           const T* v,
                           idx_t* ids,
                           const uint64_t& n,
                           const NamedArgs* args) = 0;
        virtual void open(const std::string& path) = 0;                     // open current index state and data from location specified via index_path_
        virtual bool add(const idx_t& n, T* data, idx_t* ids) = 0;          // add n to elements to index
        virtual bool remove(const idx_t& n, const idx_t* ids) = 0;          // remove n elements from index
        virtual void topk(const idx_t& nq,
                          T* query,                      // perform searches for nq queries in parallel
                          idx_t* ids,
                          T* distances,
                          double& peak_wss_mb,
                          const idx_t& k,
                          const DISTANCE_METRIC& distance_metric,
                          const float& c,
                          const NamedArgs* args) const = 0;
        virtual T* get(idx_t& n, idx_t* ids) const = 0;               // reconstruct and return the n vectors specified by ids
        virtual T* get_all() const = 0;                     // reconstruct and return all vectors
        [[nodiscard]] virtual idx_t dims() const = 0;                       // returns dimensions of index
        [[nodiscard]] virtual idx_t ntotal() const  = 0;                    // returns current number of values in index
        [[nodiscard]] virtual bool built() const = 0;
    };


} // namespace mvdb::index

#endif // INDEX_H
