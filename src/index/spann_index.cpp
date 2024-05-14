#include "spann_index.h"
#include "exception.h"

namespace mvdb::index {

    template <typename T>
    SPANNIndex<T>::SPANNIndex() {
        builder_options_ = std::make_shared<BuilderOptions>();
    }

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
        serialize_numeric<SPTAG::DimensionType>(out, builder_options_->m_dimension);
        serialize_string(out, builder_options_->m_inputFiles);
        serialize_string(out, builder_options_->m_outputFolder);
        serialize_numeric<unsigned char>(out, static_cast<const unsigned char>(builder_options_->m_indexAlgoType));
        serialize_numeric<unsigned char>(out, static_cast<const unsigned char>(builder_options_->m_inputValueType));
        serialize_string(out, builder_options_->m_builderConfigFile);
        serialize_string(out, builder_options_->m_quantizerFile);
        serialize_numeric<bool>(out, builder_options_->m_metaMapping);
        serialize_numeric<bool>(out, builder_options_->m_normalized);
        serialize_numeric<unsigned char>(out, static_cast<const unsigned char>(builder_options_->m_inputFileType));
        serialize_numeric<uint32_t>(out, builder_options_->m_threadNum);
        serialize_numeric<bool>(out, this->built_);
    }

    template <typename T>
    void SPANNIndex<T>::deserialize_(std::istream& in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != SPANN) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        this->dims_ = deserialize_numeric<idx_t>(in);
        builder_options_->m_dimension = deserialize_numeric<SPTAG::DimensionType>(in);
        builder_options_->m_inputFiles = deserialize_string(in);
        builder_options_->m_outputFolder = deserialize_string(in);
        builder_options_->m_indexAlgoType = static_cast<SPTAG::IndexAlgoType>(deserialize_numeric<unsigned char>(in));
        builder_options_->m_inputValueType = static_cast<SPTAG::VectorValueType>(deserialize_numeric<unsigned char>(in));
        builder_options_->m_builderConfigFile = deserialize_string(in);
        builder_options_->m_quantizerFile = deserialize_string(in);
        builder_options_->m_metaMapping = deserialize_numeric<bool>(in);
        builder_options_->m_normalized = deserialize_numeric<bool>(in);
        builder_options_->m_inputFileType = static_cast<SPTAG::VectorFileType>(deserialize_numeric<unsigned char>(in));
        builder_options_->m_threadNum = deserialize_numeric<uint32_t>(in);
        this->built_ = deserialize_numeric<bool>(in);
    }

    template <typename T>
    void SPANNIndex<T>::build(const idx_t &dims, const std::string& path, const std::string& initial_data_path,
                              const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) {

//        ./indexbuilder -c buildconfig.ini -d 128 -v Float -f XVEC -i sift1m/sift_base.fvecs -o sift1m_index_dir_Saved -a SPANN

        if(path.empty()) throw std::runtime_error("output path cannot be empty");

        if((!initial_data_path.empty() && initial_data_size > 0) || (initial_data_path.empty() && initial_data_size <= 0))
            throw std::runtime_error("Only exactly one of 'initial_data' and 'initial_data_path' accepted");

        if constexpr (std::is_same_v<T, int8_t>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,
                                                                     SPTAG::VectorValueType::Int8);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::Int8;
        }
        else if constexpr (std::is_same_v<T, int16_t>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,
                                                                     SPTAG::VectorValueType::Int16);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::Int16;
        }
        else if constexpr (std::is_same_v<T, uint8_t>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,
                                                                     SPTAG::VectorValueType::UInt8);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::UInt8;
        }
        else if constexpr (std::is_same_v<T, float>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,
                                                                     SPTAG::VectorValueType::Float);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::Float;
        }
        else
            throw std::runtime_error("Unsupported data SPANN Index data type");

        const auto *spann_args = dynamic_cast<const SPANNIndexNamedArgs *>(args);
        if (!spann_args)
            throw std::runtime_error("Failed to dynamic_cast from NamedArgs to SPANNIndexNamedArgs");


        this->dims_ = dims;
        builder_options_->m_dimension = static_cast<SPTAG::DimensionType>(dims);
        builder_options_->m_inputFiles = initial_data_path;
        builder_options_->m_outputFolder = path;
        builder_options_->m_indexAlgoType = SPTAG::IndexAlgoType::SPANN;
        builder_options_->m_builderConfigFile = spann_args->build_config_path;
        builder_options_->m_quantizerFile = spann_args->quantizer_path;
        builder_options_->m_metaMapping = spann_args->meta_mapping;
        builder_options_->m_normalized = spann_args->normalized;
        builder_options_->m_inputFileType = SPTAG::VectorFileType::XVEC;
        builder_options_->m_threadNum = spann_args->thread_num;


        if (!builder_options_->m_quantizerFile.empty()) {
            sptag_vector_index_->LoadQuantizer(builder_options_->m_quantizerFile);
            if (!sptag_vector_index_->m_pQuantizer)
                throw std::runtime_error("failed to open quantizer file, '" + builder_options_->m_quantizerFile + "'");
        }

        SPTAG::Helper::IniReader iniReader;
        if (!builder_options_->m_builderConfigFile.empty() && iniReader.LoadIniFile(builder_options_->m_builderConfigFile) != SPTAG::ErrorCode::Success)
            throw std::runtime_error("cannot open index configure file, '" + builder_options_->m_builderConfigFile + "'");

        std::string sections[] = { "Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index" };
        for (const auto & section : sections) {
            if (!iniReader.DoesParameterExist(section, "NumberOfThreads"))
                iniReader.SetParameter(section, "NumberOfThreads", std::to_string(builder_options_->m_threadNum));
            for (const auto& iter : iniReader.GetParameters(section)) {
                if(section == "Base" && iter.first == "indexdirectory"){
                    std::cout << "Skipping [Base][IndexDirectory] => Setting value to specified output folder = '" << builder_options_->m_outputFolder << "'" <<std::endl;
                    sptag_vector_index_->SetParameter(iter.first, builder_options_->m_outputFolder, section);
                } else {
                    sptag_vector_index_->SetParameter(iter.first, iter.second, section);
                }
            }
        }

        SPTAG::ErrorCode code;

        std::shared_ptr<SPTAG::VectorSet> vec_set;
        std::shared_ptr<SPTAG::MetadataSet> metadata_set;
        if(!initial_data_path.empty()){
            auto vectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(builder_options_);
            if (SPTAG::ErrorCode::Success != vectorReader->LoadFile(builder_options_->m_inputFiles))
                throw std::runtime_error("failed to read input file, '" + builder_options_->m_inputFiles + "'");
            vec_set = vectorReader->GetVectorSet();
            metadata_set = vectorReader->GetMetadataSet();
        } else {
            uint64_t query_size_bytes = sizeof(T) * this->dims_ * initial_data_size;
            SPTAG::ByteArray vec_set_byte_arr = SPTAG::ByteArray::Alloc(query_size_bytes);
            char* vecBuf = reinterpret_cast<char*>(vec_set_byte_arr.Data());
            memcpy(vecBuf, initial_data, query_size_bytes);
            vec_set.reset(new SPTAG::BasicVectorSet(vec_set_byte_arr,
                                                    builder_options_->m_inputValueType,
                                                    this->dims_, initial_data_size));
            metadata_set = nullptr;
        }

        code = sptag_vector_index_->BuildIndex(vec_set, metadata_set,
                                               builder_options_->m_metaMapping,
                                               builder_options_->m_normalized, true);

        if (code == SPTAG::ErrorCode::Success)
            save_(builder_options_->m_outputFolder);
        else
            throw std::runtime_error("failed to build index, '" + builder_options_->m_outputFolder + "'");
        this->built_ = true;
    }

    template<typename T>
    void SPANNIndex<T>::save_(const std::string &path) const {
        sptag_vector_index_->SaveIndex(builder_options_->m_outputFolder);
    }

    template<typename T>
    void SPANNIndex<T>::open(const std::string &path) {
        auto ret = SPTAG::VectorIndex::LoadIndex(path, sptag_vector_index_);
        if (SPTAG::ErrorCode::Success != ret || nullptr == sptag_vector_index_)
            throw std::runtime_error("failed to open index configure file in dir, '" + path + "'");
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
    void SPANNIndex<T>::topk(const idx_t &nq, T *query, const std::string& query_path,
                             const std::string& result_path, idx_t *ids, T *distances, const idx_t &k,
                             const DISTANCE_METRIC &distance_metric, const float& c) const {
//  ./indexsearcher -x sift1m_index_dir_Saved/ -d 128 -v Float -f XVEC -i sift1m/sift_query.fvecs -o outputSearch.txt -k 100

        if(query_path.empty() && query == nullptr)
            throw std::runtime_error("either query or query_path is required");

        if(!query_path.empty() && query != nullptr)
            throw std::runtime_error("Exactly one of query and query_path accepted");

        std::shared_ptr<SearcherOptions> options(new SearcherOptions);
        options->m_indexFolder = builder_options_->m_outputFolder;
        options->m_dimension = builder_options_->m_dimension;
        options->m_inputValueType = builder_options_->m_inputValueType;
        options->m_inputFileType = SPTAG::VectorFileType::XVEC;
        options->m_queryFile = query_path;
        options->m_resultFile = result_path;
//        options->m_outputformat = 0;
        options->m_K = (int)k;
        options->m_batch = 100;

        sptag_vector_index_->SetQuantizerADC(options->m_enableADC);

        SPTAG::Helper::IniReader iniReader;
        std::string sections[] = { "Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index" };
        for (const auto & section : sections) {
            if (!iniReader.DoesParameterExist(section, "NumberOfThreads"))
                iniReader.SetParameter(section, "NumberOfThreads", std::to_string(options->m_threadNum));
            for (const auto& iter : iniReader.GetParameters(section))
                sptag_vector_index_->SetParameter(iter.first, iter.second, section);
        }
        sptag_vector_index_->UpdateIndex();

        std::shared_ptr<SPTAG::VectorSet> vec_set = nullptr;
        std::shared_ptr<SPTAG::MetadataSet> meta_set(nullptr);
        if(options->m_queryFile.empty()) {
            uint64_t query_size_bytes = sizeof(T) * this->dims_ * nq;
            SPTAG::ByteArray vec_set_byte_arr = SPTAG::ByteArray::Alloc(query_size_bytes);
            char *vecBuf = reinterpret_cast<char *>(vec_set_byte_arr.Data());
            memcpy(vecBuf, query, query_size_bytes);
            vec_set.reset(new SPTAG::BasicVectorSet(vec_set_byte_arr, builder_options_->m_inputValueType, this->dims_, nq));
        }

        switch (options->m_inputValueType) {
            #define DefineVectorValueType(Name, Type) \
            case SPTAG::VectorValueType::Name: \
            Process<Type>(options, *(sptag_vector_index_.get()), ids, distances, vec_set, meta_set); \
            break; \

            #include <inc/Core/DefinitionList.h>
            #undef DefineVectorValueType

            default: break;
        }
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
        return this->dims_;
    }

    template<typename T>
    idx_t SPANNIndex<T>::ntotal() const {
//        sptag_vector_index_->
        return 0;
    }

    template <typename T>
    bool SPANNIndex<T>::built() const {
        return this->built_;
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