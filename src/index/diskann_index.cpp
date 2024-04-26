#include "diskann_index.h"
#include "filesystem.h"

namespace mvdb::index {

    template <typename T>
    IndexType DiskANNIndex<T>::type() const {
        return DISKANN;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const DiskANNIndex<T>& obj){
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const DiskANNIndex<T>* obj){
        return os << "*(" << *obj << ")";
    }

    template <typename T>
    void DiskANNIndex<T>::serialize_(std::ostream& out) const {
//        serialize_numeric<unsigned char>(out, type());
//        serialize_numeric<idx_t>(out, this->dims_);
    }

    template <typename T>
    void DiskANNIndex<T>::deserialize_(std::istream& in) {
//        auto type = deserialize_numeric<unsigned char>(in);
//        if (type != FAISS_FLAT) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
//        this->dims_ = deserialize_numeric<idx_t>(in);
    }

    template <typename T>
    void DiskANNIndex<T>::build(const mvdb::idx_t &dims, const std::string &path) {
        this->dims_ = dims;

        if(path.empty())
            throw std::runtime_error("index path cannot be empty");

        if(fs::exists(path))
            throw std::runtime_error("'" + path + "' is invalid. make sure no file or dir exists at that location");


        std::string data_path = path + "_data";
        std::string index_path_prefix = path + "_index";

        bool use_opq = true;

//        diskann::build_disk_index<float>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
//                                         metric, use_opq, codebook_prefix, use_filters, label_file,
//                                         universal_label, filter_threshold, Lf);

    }

    template <typename T>
    void DiskANNIndex<T>::save_(const std::string &path) const {}

}