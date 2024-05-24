#include "annoy_index.h"
#include "filesystem.h"
#include "utils.h"

#include <type_traits>

#include <omp.h>
#include <cstring>
#include <sys/resource.h>

namespace mvdb::index {

    template <typename T>
    void MVDBAnnoyIndex<T>::build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) {
        if (path.empty())
            throw std::invalid_argument("Index path cannot be empty.");
        if (fs::exists(path))
            throw std::invalid_argument("Path '" + path + "' already exists. Please specify a new path.");
        if (!args)
            throw std::invalid_argument("NamedArgs pointer cannot be null.");

        const auto *annoy_args = dynamic_cast<const AnnoyIndexNamedArgs *>(args);
        if (!annoy_args)
            throw std::runtime_error("Failed to dynamic_cast from NamedArgs to AnnoyIndexNamedArgs");

        std::cout << "Building MVDBAnnoyIndex @ " << path << std::endl;
        std::cout << "Dimensions = " << dims << " | Number of Trees = " << annoy_args->n_trees << " | Number of Threads = " << annoy_args->n_threads << std::endl;

        this->dims_ = dims;
        this->n_trees = annoy_args->n_trees;
        this->n_threads = annoy_args->n_threads;

        #ifdef ANNOYLIB_MULTITHREADED_BUILD
        annoy_index_ = std::make_unique<Annoy::AnnoyIndex<int, T, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>>(dims);
        #else
        annoy_index_ = std::make_unique<Annoy::AnnoyIndex<int, T, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>>(dims);
        #endif

        if (!initial_data_path.empty()) {
            std::vector<T> data;
            std::vector<size_t> start_indexes;

            int num_vecs = 0;
            if(std::is_same_v<T, float>)
                num_vecs = xvecs_num_vecs<float>(initial_data_path);
            else
                num_vecs = xvecs_num_vecs<int32_t>(initial_data_path);
            std::cout << "loading file " << initial_data_path << std::endl;
            read_xvecs<T>(initial_data_path, data, start_indexes, num_vecs);
            std::cout << "adding elements to Annoy index..." << std::endl;
            for (int i = 0; i < num_vecs; i++)
                annoy_index_->add_item(i, data.data() + i * dims);
            std::cout << "finished elements to Annoy index" << std::endl;
            data.clear();
            data.shrink_to_fit();
        } else if (initial_data_size > 0) {
            if (!initial_data)
                throw std::invalid_argument("Initial data pointer cannot be null when size is specified.");
            std::cout << "adding elements to Annoy index..." << std::endl;
            for (size_t i = 0; i < initial_data_size; i++) {
                annoy_index_->add_item(i, initial_data + i * dims);
            }
            std::cout << "finished elements to Annoy index" << std::endl;
        }
        annoy_index_->build(this->n_trees, this->n_threads);
        this->save_(path);
        this->built_ = true;
    }

    template<typename T>
    void MVDBAnnoyIndex<T>::save_(const std::string &path) const {
        annoy_index_->save(path.c_str());
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const MVDBAnnoyIndex<T> &obj) {
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl
                    << "n_trees: " << obj.n_trees << std::endl
                    << "n_threads: " << obj.n_threads << std::endl;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &os, const MVDBAnnoyIndex<T> *obj) {
        return os << "*(" << *obj << ")";
    }

    template<typename T>
    void MVDBAnnoyIndex<T>::serialize_(std::ostream &out) const {
        serialize_numeric<unsigned char>(out, type());
        serialize_numeric<idx_t>(out, this->dims_);
        serialize_numeric<int>(out, this->n_trees);
        serialize_numeric<int>(out, this->n_threads);
        serialize_numeric<bool>(out, this->built_);
    }

    template<typename T>
    void MVDBAnnoyIndex<T>::deserialize_(std::istream &in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != mvdb::index::IndexType::ANNOY) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        this->dims_ = deserialize_numeric<idx_t>(in);
        this->n_trees = deserialize_numeric<int>(in);
        this->n_threads = deserialize_numeric<int>(in);
        this->built_ = deserialize_numeric<bool>(in);
    }

    template<typename T>
    IndexType MVDBAnnoyIndex<T>::type() const {
        return mvdb::index::IndexType::ANNOY;
    }

    template<typename T>
    void MVDBAnnoyIndex<T>::open(const std::string &path) {
        #ifdef ANNOYLIB_MULTITHREADED_BUILD
                annoy_index_ = std::make_unique<Annoy::AnnoyIndex<int, T, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>>(this->dims_);
        #else
                annoy_index_ = std::make_unique<Annoy::AnnoyIndex<int, T, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>>(this->dims_);
        #endif
        if (!annoy_index_->load(path.c_str()))
            throw std::runtime_error("Failed to load index file, '" + path +"'\n");
    }

    template<typename T>
    bool MVDBAnnoyIndex<T>::add(const idx_t &n, T *data, idx_t *ids) {
        return false;
    }

    template<typename T>
    bool MVDBAnnoyIndex<T>::remove(const idx_t &n, const idx_t *ids) {
        return false;
    }

    template<typename T>
    void MVDBAnnoyIndex<T>::topk(const idx_t &nq, T *query, const std::string& query_path,
                                 const std::string& result_path, idx_t *ids, T *distances,
                                 double& peak_wss_mb, const idx_t &k, const DISTANCE_METRIC &distance_metric,
                                 const float &c) const {

        if(!query_path.empty())
            throw std::runtime_error("Query file with MVDBAnnoyIndex topk not supported...yet");

        if(!result_path.empty())
            throw std::runtime_error("Result file with MVDBAnnoyIndex topk not supported...yet");

        #pragma omp parallel
        {
            std::vector<int> closest_ids;           // DO NOT pre-allocate memory for Annoy recepticles!
            std::vector<T> closest_distances;       // Annoy's search methods will ignore all pre-allcoated memory and allocate it's own memory right after
            #pragma omp parallel for schedule(dynamic) if (nq >= 10)
            for (idx_t i = 0; i < nq; i++) {
                annoy_index_->get_nns_by_vector(query + i * this->dims_, k, -1, &closest_ids, &closest_distances);
                for (int j = 0; j < k; j++) {
                    ids[i * k + j] = closest_ids[j];
                    distances[i * k + j] = closest_distances[j];
                }
            }
        };

        #ifndef _MSC_VER
        struct rusage rusage{};
        getrusage(RUSAGE_SELF, &rusage);
        double peakWSS = (double)(rusage.ru_maxrss) / (double)1024;
        #else
        PROCESS_MEMORY_COUNTERS pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
        unsigned long long peakWSS = pmc.PeakWorkingSetSize / 1000000000;
        #endif
        peak_wss_mb = peakWSS;
    }


    template<typename T>
    T *MVDBAnnoyIndex<T>::get(idx_t &n, idx_t *keys) const {
        return nullptr;
    }

    template<typename T>
    T *MVDBAnnoyIndex<T>::get_all() const {
        return nullptr;
//        std::unique_ptr<T[]> v(new T[this->dims()]);
//        for (int i = 0; i < k; ++i) {
//            annoy_index_->get_item(ids[i], v.get());
//            std::cout << "Vector found => " << ids[i] << " =>  ";
//            std::cout << "Distance => " << distances[i] << " =>  ";
//            for (int j = 0; j < this->dims(); ++j) {
//                std::cout << v[j] << " ";
//            }
//            std::cout << std::endl;
//        }
    }

    template <typename T>
    idx_t MVDBAnnoyIndex<T>::dims() const {
        return this->dims_;
    }

    template <typename T>
    idx_t MVDBAnnoyIndex<T>::ntotal() const {
        return this->annoy_index_->get_n_items();
    }

    template <typename T>
    bool MVDBAnnoyIndex<T>::built() const {
        return this->built_;
    }

    template class MVDBAnnoyIndex<int8_t>;
    template class MVDBAnnoyIndex<int16_t>;
    template class MVDBAnnoyIndex<uint8_t>;
    template class MVDBAnnoyIndex<float>;
}