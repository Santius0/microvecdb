#include "faiss_flat_index.h"
#include "exception.h"
#include <iostream>
#include <filesystem>

namespace mvdb {

    IndexType FaissFlatIndex::type() const {
        return FAISS_FLAT;
    }

    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex& obj){
        return os   << "index_path_: " << obj.index_path_ << std::endl
                    << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex* obj){
        return os << "*(" << *obj << ")";
    }

    // this should always save the index type along with all other data
    void FaissFlatIndex::serialize(std::ostream& out) const {
        serialize_numeric<uint64_t>(out, static_cast<uint64_t>(type()));
        serialize_numeric<uint64_t>(out, dims_);
        serialize_string(out, index_path_);
        save();
    }

    // this should only be run when you have already read the index type and know you're creating this specific index type
    void FaissFlatIndex::deserialize(std::istream& in) {
        dims_ = deserialize_numeric<uint64_t>(in);
        index_path_ = deserialize_string(in);
    }

    FaissFlatIndex::FaissFlatIndex(const std::string& index_path, const idx_t& dims):
    Index(index_path, dims) {}

    bool FaissFlatIndex::add(const idx_t& n, value_t* data, idx_t* ids) {
        if(n <= 0) return false;
        if(data == nullptr || ids == nullptr) return false;
        try {
            if(!is_open_) throw std::runtime_error("FaissFlatIndex not open. Use index->open() before running add");
            if(!faiss_index_) throw std::runtime_error("faiss_index_ == nullptr. Use index->open() before running add");
            for(int i = 0; i < n; i++)
                ids[i] = faiss_index_->ntotal + i;
            faiss_index_->add(static_cast<long>(n), static_cast<float*>(data));
            save();
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

    void FaissFlatIndex::save() const {
        if(this->is_open_){
            faiss::write_index(faiss_index_.get(), index_path_.c_str());
        }
    }

    void FaissFlatIndex::load() {
//        if(this->is_open_){
//            faiss::write_index(faiss_index_.get(), this->index_path_.c_str());
//        }
    }

    void FaissFlatIndex::open() {
        if(!is_open_) {
            if(std::filesystem::exists(index_path_)) {
                if (std::filesystem::is_directory(index_path_))
                    throw std::runtime_error(index_path_ + " already exists as directory");
//                faiss_index_.reset(faiss::read_index(index_path_.c_str(), faiss::IO_FLAG_MMAP));
                faiss_index_.reset(faiss::read_index(this->index_path_.c_str()));   // load the faiss index from file
            }
            else faiss_index_ = std::make_unique<faiss::IndexFlatL2>(this->dims_); // initialise internal faiss index
            is_open_ = true;
        }
    }

    void FaissFlatIndex::close(bool force) {
        faiss_index_->reset();
        this->is_open_ = false;
    }

    /** query n vectors of dimension d to the index.
     *
     * return at most k vectors. If there are not enough results for a
     * query, the result array is padded with -1s.
     *
     * @param nq            number of query vectors
     * @param query         input vectors to search, size n * d
     * @param ids           output labels of the NNs, size n*k
     * @param distances     output pairwise distances, size n*k
     * @param k             number of extracted vectors
     */
    void FaissFlatIndex::search(const idx_t& nq, value_t* query, idx_t* ids, value_t* distances, const idx_t& k) const{
        faiss_index_->search(static_cast<long>(nq), static_cast<float*>(query), k, static_cast<float*>(distances), reinterpret_cast<faiss::idx_t *>(ids));
    }

    faiss::Index* FaissFlatIndex::faiss_index(){
        return faiss_index_.get();
    }

    value_t* FaissFlatIndex::get(idx_t& n, idx_t* keys) const {
        if(!this->is_open_)
            throw std::runtime_error("get failed => faiss_index has not been properly opened");

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

        if(keys == nullptr) n = faiss_index_->ntotal;
        auto* reconstructed_vec = new float[n * this->dims_];
        for(faiss::idx_t i = 0; i < n; i++){
            faiss::idx_t key = (keys == nullptr ? i : static_cast<faiss::idx_t>(keys[i]));
            faiss_index_->reconstruct(key, reconstructed_vec + (i * this->dims_));
        }
        return reconstructed_vec;
    }

    bool FaissFlatIndex::is_open() const {
        return is_open_;
    }

    idx_t FaissFlatIndex::dims() const {
        return dims_;
    }

    idx_t FaissFlatIndex::ntotal() const {
        return ntotal_;
    }
} // namespace mvdb
