#include "faiss_flat_index.h"
#include "exception.h"
#include "filesystem.h"

#include <iostream>
#include <fstream>

namespace mvdb {

    IndexType FaissFlatIndex::type() const {
        return FAISS_FLAT;
    }

    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex& obj){
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex* obj){
        return os << "*(" << *obj << ")";
    }

    // this should always save the index type along with all other data
    void FaissFlatIndex::serialize(std::ostream& out) const {
        serialize_numeric<unsigned char>(out, type());
        serialize_numeric<idx_t>(out, dims_);
    }

    // this should only be run when you have already read the index type and know you're creating this specific index type
    void FaissFlatIndex::deserialize(std::istream& in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != FAISS_FLAT) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        dims_ = deserialize_numeric<idx_t>(in);
    }

    FaissFlatIndex::FaissFlatIndex(const idx_t& dims, const std::string& path): Index(dims, path) {
        faiss_index_ = std::make_unique<faiss::IndexFlatL2>(this->dims_);
        if(!path.empty() && fs::exists(path) && !fs::is_directory(path)) {
            std::string faiss_path = path + "_faiss";
            if(!faiss_path.empty() && fs::exists(faiss_path) && !fs::is_directory(faiss_path)) {
                load(path);
            } else {
                throw std::runtime_error("'" + faiss_path + "' does not exist");
            }
        }
    }

    bool FaissFlatIndex::add(const idx_t& n, value_t* data, idx_t* ids) {
        if(n <= 0) return false;
        if(data == nullptr) return false;
        try {
            if(!faiss_index_) throw std::runtime_error("faiss_index_ == nullptr. Use index->open() before running add");
            if(ids) {
                for (int i = 0; i < n; i++) ids[i] = faiss_index_->ntotal + i;
            }
            faiss_index_->add(static_cast<long>(n), static_cast<float*>(data));
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

//    bool FaissFlatIndex::remove(const size_t& n, const faiss::IDSelector& ids) const {
    bool FaissFlatIndex::remove(const idx_t& n, const idx_t* ids) {
        throw not_implemented("FaissFlatIndex<T, D>::remove(const idx_t& n, idx_t* ids) const not implemented");
//        try {
//            faiss_index_->remove_ids(ids);
//            save();
//            return true;
//        } catch (const std::exception& e) {
//            std::cerr << "Error removing data from index: " << e.what() << std::endl;
//            return false;
//        }
    }

    void FaissFlatIndex::save(const std::string& path) const {
        std::ofstream out_file(path, std::ios::binary | std::ios::out);
        serialize(out_file);
        out_file.close();
        faiss::write_index(faiss_index_.get(), (path + "_faiss").c_str());
    }

    void FaissFlatIndex::load(const std::string& path) {
        std::ifstream in_file(path, std::ios::binary | std::ios::in);
        deserialize(in_file);
        in_file.close();
        faiss_index_.reset(faiss::read_index((path + "_faiss").c_str()));
    }

    void FaissFlatIndex::search(const idx_t& nq, value_t* query, idx_t* ids, value_t* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const{
        faiss_index_->search(static_cast<long>(nq), static_cast<float*>(query), k, static_cast<float*>(distances), reinterpret_cast<faiss::idx_t *>(ids));
    }

    faiss::Index* FaissFlatIndex::faiss_index(){
        return faiss_index_.get();
    }

    value_t* FaissFlatIndex::get(idx_t& n, idx_t* keys) const {
        if (faiss_index_->is_trained) {
            if(keys != nullptr){
                for(size_t i = 0; i < n; i++) {
                    if(faiss_index_->ntotal <= keys[i]){
                        std::cerr << "ID '" << keys[i] << "' is out of index range." << std::endl;
                        return nullptr;
                    }
                }
            }
        }
        else {
            std::cerr << "Index does not support reconstruction." << std::endl;
            return nullptr;
        }
    }

    value_t* FaissFlatIndex::get_all() const {
        idx_t n = faiss_index_->ntotal;
        auto *keys = new idx_t[n];
        auto* reconstructed_vec = new float[n * this->dims_];
        for(faiss::idx_t i = 0; i < n; i++){
            auto key = static_cast<faiss::idx_t>(keys[i]);
            faiss_index_->reconstruct(key, reconstructed_vec + (i * this->dims_));
        }
        return reconstructed_vec;
    }

    idx_t FaissFlatIndex::dims() const {
        return dims_;
    }

    idx_t FaissFlatIndex::ntotal() const {
        return ntotal_;
    }

    value_t* FaissFlatIndex::index() const {
        return get_all();
    }

} // namespace mvdb
