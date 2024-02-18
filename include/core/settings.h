#ifndef MICROVECDB_SETTINGS_H
#define MICROVECDB_SETTINGS_H

#include "serializable.h"

namespace mvdb {

    class Settings final : public Serializable {
        // PCA settings
        bool pca_ = true;
        float pca_var_ = 90.0f;

        // product quantization settings
        bool quantize_ = true;
        int pq_nprobe_ = 4;
        int pq_nlist_ = 32;

        // rocksdb settings
        int compression_alg = 0; // TODO: create enum for all different compression types
//        int column_family_thing;

    protected:
        void serialize(std::ostream &out) const override {};
        void deserialize(std::istream &in) override {};
    public:
        Settings(bool pca, float pca_var, bool quantize, int pq_nprobe, int pq_nlist) :
        pca_(pca), pca_var_(pca_var), quantize_(quantize), pq_nprobe_(pq_nprobe), pq_nlist_(pq_nlist) {}
        ~Settings() override = default;
        void optimise(){

        };
    };

}

#endif //MICROVECDB_SETTINGS_H
