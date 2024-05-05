#include "annoy_index.h"

namespace mvdb::index {

    void build(const idx_t& dims, const std::string& path, const NamedArgs& args) {
        if(const auto * spann_args = dynamic_cast<const AnnoyIndexNamedArgs*>(&args)){

        } else {
            throw std::runtime_error("failed to dynamic_cast from NamedArgs to AnnoyIndexNamedArgs");
        }
    }

}