#include "spann_index.h"

namespace mvdb::index {

    template <typename T>
    IndexType SPANNIndex<T>::type() const {
        return IndexType::SPANN;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>& obj){
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>* obj){
        return os << "*(" << *obj << ")";
    }

    template <typename T>
    void SPANNIndex<T>::serialize_(std::ostream& out) const {
        serialize_numeric<unsigned char>(out, type());
        serialize_numeric<idx_t>(out, this->dims_);
    }

    template <typename T>
    void SPANNIndex<T>::deserialize_(std::istream& in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != SPANN) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        this->dims_ = deserialize_numeric<idx_t>(in);
    }

    template <typename T>
    void SPANNIndex<T>::build(const idx_t& dims, const std::string& path, const NamedArgs& args) {
        if(const auto * spann_args = dynamic_cast<const SPANNIndexNamedArgs*>(&args)){

        } else {
            throw std::runtime_error("failed to dynamic_cast from NamedArgs to SPANNIndexNamedArgs");
        }
    }

}