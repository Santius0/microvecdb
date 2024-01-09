#include "index.h"
#include <filesystem>

namespace mvdb {

    Index::Index(std::string index_path, const uint64_t &dims, const IndexType &index_type):
    index_path_(std::move(index_path)), dims_(dims), index_type_(index_type){}

     void Index::serialize(std::ostream& out) const {
        serialize_string(out, index_path_);
        serialize_numeric<uint64_t>(out, dims_);
        serialize_numeric<uint64_t>(out, static_cast<uint64_t>(index_type_));
    }

    void Index::deserialize(std::istream& in) {
        index_path_ = deserialize_string(in);
        dims_ = deserialize_numeric<uint64_t>(in);
        index_type_ = static_cast<IndexType>(deserialize_numeric<uint64_t>(in));
    }

    void Index::open() {
        if(!is_open_){
            if(std::filesystem::exists(index_path_)) load();
            else init();
            is_open_ = true;
        }
    }

    bool Index::is_open() const {
        return is_open_;
    }

    uint64_t Index::dims(){
        return dims_;
    }

}