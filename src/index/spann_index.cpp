#include "spann_index.h"

#include "Helper/VectorSetReader.h"
#include "Core/VectorIndex.h"
#include "Core/Common.h"
#include "Helper/SimpleIniReader.h"

#include <memory>
#include <iostream>
#include <Core/Common/DistanceUtils.h>

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
    void SPANNIndex<T>::build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) {
        // Hard-coded options setup
        BuilderOptions options;
        options.m_inputFiles = "path_to_input_vectors.txt"; // Specify the input file path
        options.m_outputFolder = "path_to_output_folder";   // Specify the output folder path
        options.m_indexAlgoType = SPTAG::IndexAlgoType::KDT;       // Specify the index algorithm type
        options.m_builderConfigFile = "path_to_config.ini"; // Specify the configuration file path
        options.m_quantizerFile = "path_to_quantizer_file"; // Specify the quantizer file path
        options.m_metaMapping = true;                       // Enable or disable metadata mapping

        // Load configuration from ini file
        SPTAG::Helper::IniReader iniReader;
        if (iniReader.LoadIniFile(options.m_builderConfigFile) != SPTAG::ErrorCode::Success) {
            std::cerr << "Cannot open index configure file!\n";
            return;
        }

        // Create index builder instance based on the algorithm type
        auto indexBuilder = SPTAG::VectorIndex::CreateInstance(options.m_indexAlgoType, options.m_inputValueType);
        if (!options.m_quantizerFile.empty()) {
            indexBuilder->LoadQuantizer(options.m_quantizerFile);
            if (!indexBuilder->m_pQuantizer) {
                std::cerr << "Failed to load quantizer.\n";
                return;
            }
        }

        // Set additional parameters from the ini file
        std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index"};
        for (const auto& section : sections) {
            for (const auto& iter : iniReader.GetParameters(section)) {
                indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str(), section);
            }
        }

        // Load vectors and build index
        SPTAG::ErrorCode code;
        std::shared_ptr<SPTAG::VectorSet> vecset;
        auto vectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(&options);
        if (vectorReader->LoadFile(options.m_inputFiles) != SPTAG::ErrorCode::Success) {
            std::cerr << "Failed to read input file.\n";
            return;
        }
        vecset = vectorReader->GetVectorSet();
        code = indexBuilder->BuildIndex(vecset, vectorReader->GetMetadataSet(), options.m_metaMapping, options.m_normalized, true);

        // Check for success and save the index
        if (code == SPTAG::ErrorCode::Success) {
            indexBuilder->SaveIndex(options.m_outputFolder);
        } else {
            std::cerr << "Failed to build index.\n";
            return;
        }
    }

    template<typename T>
    void SPANNIndex<T>::save_(const std::string &path) const {

    }

    template<typename T>
    void SPANNIndex<T>::open(const std::string &path) {

    }

    template<typename T>
    bool SPANNIndex<T>::add(const idx_t &n, T *data, idx_t *ids) {
        return false;
    }

    template<typename T>
    bool SPANNIndex<T>::remove(const idx_t &n, const idx_t *ids) {
        return false;
    }

    template<typename T>
    void SPANNIndex<T>::topk(const idx_t &nq, T *query, idx_t *ids, T *distances, const idx_t &k,
                               const DISTANCE_METRIC &distance_metric, const float& c) const {

    }

    template<typename T>
    T *SPANNIndex<T>::get(idx_t &n, idx_t *keys) const {
        return nullptr;
    }

    template<typename T>
    T *SPANNIndex<T>::get_all() const {
        return nullptr;
    }

    template<typename T>
    idx_t SPANNIndex<T>::dims() const {
        return 0;
    }

    template<typename T>
    idx_t SPANNIndex<T>::ntotal() const {
        return 0;
    }

    template class SPANNIndex<int8_t>;
    template class SPANNIndex<int16_t>;
    template class SPANNIndex<int32_t>;
    template class SPANNIndex<int64_t>;
    template class SPANNIndex<uint8_t>;
    template class SPANNIndex<uint16_t>;
    template class SPANNIndex<uint32_t>;
    template class SPANNIndex<uint64_t>;
    template class SPANNIndex<float>;
    template class SPANNIndex<double>;
}