#include "vector_index.hpp"
#include "utils.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <utility>

namespace mvdb {

    VectorIndexMetadata::VectorIndexMetadata(std::string indexFilePath, const uint64_t& indexDimensions,
        const VectorIndexType& indexType): indexFilePath(std::move(indexFilePath)), indexDimensions(indexDimensions),
    indexType(indexType) {}

    void VectorIndexMetadata::serialize(std::ostream& out) const {
        serializeString(out, indexFilePath);
        serializeUInt64T(out, indexDimensions);
        serializeUInt64T(out, static_cast<uint64_t>(indexType));
    }

    void VectorIndexMetadata::deserialize(std::istream& in) {
        indexFilePath = deserializeString(in);
        indexDimensions = deserializeUInt64T(in);
        indexType = static_cast<VectorIndexType>(deserializeUInt64T(in));
    }

    VectorIndex::VectorIndex(const VectorIndexMetadata& metadata) {
        indexFilePath = metadata.indexFilePath;
        if(std::filesystem::exists(metadata.indexFilePath)) load();
        else {
            switch (metadata.indexType) {
                case VectorIndexType::IVF: {
                    constexpr int nlist = 100;
                    faiss::IndexFlatL2 quantizer(static_cast<long>(metadata.indexDimensions));
                    faissIndex = std::make_unique<faiss::IndexIVFFlat>(&quantizer, metadata.indexDimensions, nlist, faiss::METRIC_L2);
                    break;
                }
                default: {
                    faissIndex = std::make_unique<faiss::IndexFlatL2>(metadata.indexDimensions);
                    break;
                }
            }
        }
    }

    // TODO: only save if something changed
    VectorIndex::~VectorIndex() {
        if(faissIndex != nullptr) save();
    }

    bool VectorIndex::add(const size_t& n, const float* data) const {
        try {
            faissIndex->add(static_cast<long>(n), data);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
    }

    bool VectorIndex::remove(const size_t& n, const faiss::IDSelector& ids) const {
        try {
            faissIndex->remove_ids(ids);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error removing data from index: " << e.what() << std::endl;
            return false;
        }
    }

    void VectorIndex::save() const {
        faiss::write_index(faissIndex.get(), indexFilePath.c_str());
    }

    void VectorIndex::load() {
        faissIndex.reset(faiss::read_index(indexFilePath.c_str()));
    }

} // namespace mvdb
