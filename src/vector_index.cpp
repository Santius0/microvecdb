#include "vector_index.hpp"
#include "constants.hpp"
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

    VectorIndex::VectorIndex(std::string  name, std::string  dir,
         VectorIndexType type, uint64_t dims)
        : name(std::move(name)), dir(std::move(dir)), dims(dims), type(type) {

        switch (type) {
            case VectorIndexType::IVF: {
                constexpr int nlist = 100;
                faiss::IndexFlatL2 quantizer(static_cast<long>(dims));
                faissIndex = std::make_unique<faiss::IndexIVFFlat>(&quantizer, dims, nlist, faiss::METRIC_L2);
                break;
            }
            default: {
                faissIndex = std::make_unique<faiss::IndexFlatL2>(dims);
                break;
            }
        }
    }

    VectorIndex* VectorIndex::create(const std::string& name, const std::string& dir,
         const VectorIndexType& type, const uint64_t& dims) {
        return new VectorIndex(name, dir, type, dims);
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

    bool VectorIndex::save() const {
        try {
            const std::string index_path = dir + "/" + name + INDEX_EXT;
            const std::string index_meta_path = dir + "/" + name + INDEX_META_EXT;

            if(!std::filesystem::exists(std::filesystem::path(this->dir)))
                if(std::error_code ec; !std::filesystem::create_directory(dir, ec))
                    std::cerr << "Error creating directory: " << ec.message() << '\n';

            faiss::write_index(faissIndex.get(), index_path.c_str());

            std::ofstream file(index_meta_path, std::ios::binary);
            if (!file) {
                std::cerr << "Error opening file for writing vector_index \"" << index_meta_path << "\"\n";
                return false;
            }
            size_t name_len = this->name.size() + 1, dir_len = this->dir.size() + 1;
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(reinterpret_cast<const char*>(&dir_len), sizeof(dir_len));
            file.write(this->name.data(), static_cast<long>(name_len));
            file.write(this->dir.data(), static_cast<long>(dir_len));
            file.write(reinterpret_cast<const char*>(&this->dims), sizeof(this->dims));
            file.write(reinterpret_cast<const char*>(&this->type), sizeof(this->type));

            if (!file) {
                std::cerr << "Error writing to file \"" << index_meta_path << "\"\n";
                return false;
            }
            file.close();

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error saving index: " << e.what() << std::endl;
            return false;
        }
    }

     VectorIndex* VectorIndex::load(const std::string& name, const std::string& dir) {
        try {
            const std::string index_path = dir + "/" + name + INDEX_EXT;
            const std::string index_meta_path = dir + "/" + name + INDEX_META_EXT;
            auto* vi = new VectorIndex(name, dir, VectorIndexType::FLAT, 0); // Temporary type and dims
            vi->faissIndex.reset(faiss::read_index(index_path.c_str()));

            std::ifstream file(index_meta_path, std::ios::binary);
            if (!file) {
                std::cerr << "Error opening file for reading vector_index \"" << index_meta_path << "\"\n";
                return nullptr;
            }

            size_t name_len, dir_len;
            std::string name_buffer, dir_buffer;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            file.read(reinterpret_cast<char*>(&dir_len), sizeof(dir_len));
            vi->name.resize(name_len);
            vi->dir.resize(name_len);
            file.read(vi->name.data(), static_cast<long>(name_len));
            file.read(vi->dir.data(), static_cast<long>(dir_len));
            file.read(reinterpret_cast<char*>(&vi->dims), sizeof(vi->dims));
            file.read(reinterpret_cast<char*>(&vi->type), sizeof(vi->type));

            if (!file) {
                std::cerr << "Error reading from file \"" << index_meta_path << "\"\n";
                return nullptr;
            }
            file.close();
            return vi;
        } catch (const std::exception& e) {
            std::cerr << "Error loading index: " << e.what() << std::endl;
            return nullptr;
        }
    }

} // namespace mvdb
