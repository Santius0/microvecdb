#include "annoy_index.h"

namespace mvdb::index {

    template <typename T>
    void AnnoyIndex<T>::build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) {
        if(args) {
            if (const auto * spann_args = dynamic_cast<const AnnoyIndexNamedArgs *>(args)) {

            } else {
                throw std::runtime_error("failed to dynamic_cast from NamedArgs to AnnoyIndexNamedArgs");
            }
        }

        std::cout << "BUILDING AnnoyIndex at " << path << std::endl;
    }

    template class AnnoyIndex<int8_t>;
    template class AnnoyIndex<int16_t>;
    template class AnnoyIndex<int32_t>;
    template class AnnoyIndex<int64_t>;
    template class AnnoyIndex<uint8_t>;
    template class AnnoyIndex<uint16_t>;
    template class AnnoyIndex<uint32_t>;
    template class AnnoyIndex<uint64_t>;
    template class AnnoyIndex<float>;
    template class AnnoyIndex<double>;
}