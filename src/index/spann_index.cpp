#include "spann_index.h"
#include "exception.h"
//#include "filesystem.h"


namespace mvdb::index {

    std::ostream& operator<<(std::ostream& os, const SPANNIndexNamedArgs& args) {
        os << "SPANNIndexNamedArgs {"
           << "\n  build_config_path: " << args.build_config_path
           << "\n  truth_path: " << args.truth_path
           << "\n  quantizer_path: " << args.quantizer_path
           << "\n  meta_mapping: " << std::boolalpha << args.meta_mapping
           << "\n  normalized: " << std::boolalpha << args.normalized
           << "\n  thread_num: " << args.thread_num
           << "\n  batch_size: " << args.batch_size
           << "\n  BKTKmeansK: " << args.BKTKmeansK
           << "\n  Samples: " << args.Samples
           << "\n  TPTNumber: " << args.TPTNumber
           << "\n  RefineIterations: " << args.RefineIterations
           << "\n  NeighborhoodSize: " << args.NeighborhoodSize
           << "\n  CEF: " << args.CEF
           << "\n  MaxCheckForRefineGraph: " << args.MaxCheckForRefineGraph
           << "\n  NumberOfInitialDynamicPivots: " << args.NumberOfInitialDynamicPivots
           << "\n  GraphNeighborhoodScale: " << args.GraphNeighborhoodScale
           << "\n  NumberOfOtherDynamicPivots: " << args.NumberOfOtherDynamicPivots
           << "\n}";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const SPANNIndexNamedArgs* args) {
        return os << *args;
    }

    const SPANNIndexNamedArgs* parse_spann_named_args(const NamedArgs* args) {
        const auto *spann_args = dynamic_cast<const SPANNIndexNamedArgs *>(args);
        if (!spann_args)
            throw std::runtime_error("Failed to dynamic_cast from NamedArgs to SPANNIndexNamedArgs");
        return spann_args;
    }

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
    void SPANNIndex<T>::build(const idx_t &dims,
                              const std::string& path,
                              const T* v,
                              idx_t* ids,
                              const uint64_t& n,
                              const NamedArgs* args) {

//        if (path.empty())
//            throw std::invalid_argument("Index path cannot be empty.");
//        if (fs::exists(path))
//            throw std::invalid_argument("Path '" + path + "' already exists. Please specify a new path.");
//        if (!args)
//            throw std::invalid_argument("NamedArgs pointer cannot be null.");

        if constexpr (std::is_same_v<T, int8_t>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,SPTAG::VectorValueType::Int8);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::Int8;
        }
        else if constexpr (std::is_same_v<T, int16_t>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,SPTAG::VectorValueType::Int16);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::Int16;
        }
        else if constexpr (std::is_same_v<T, uint8_t>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,SPTAG::VectorValueType::UInt8);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::UInt8;
        }
        else if constexpr (std::is_same_v<T, float>) {
            sptag_vector_index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN,SPTAG::VectorValueType::Float);
            builder_options_->m_inputValueType = SPTAG::VectorValueType::Float;
        }
        else throw std::runtime_error("Unsupported data SPANN Index data type");

        const auto *spann_args = parse_spann_named_args(args);

        this->dims_ = dims;
        builder_options_->m_dimension = static_cast<SPTAG::DimensionType>(dims);
        builder_options_->m_inputFiles = "";
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
        std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index"};

        if(builder_options_->m_builderConfigFile.empty())
            throw std::runtime_error("no ini file, '" + builder_options_->m_builderConfigFile + "'");

        if (iniReader.LoadIniFile(builder_options_->m_builderConfigFile) != SPTAG::ErrorCode::Success)
            throw std::runtime_error("cannot open index configure file, '" + builder_options_->m_builderConfigFile + "'");

        if constexpr (std::is_same_v<T, int8_t>)
            iniReader.SetParameter("Base", "ValueType", "Int8");
        else if constexpr (std::is_same_v<T, float>)
            iniReader.SetParameter("Base", "ValueType", "Float");

        iniReader.SetParameter("Base", "Dim", std::to_string(dims));

        if(!spann_args->truth_path.empty()) iniReader.SetParameter("Base", "TruthPath", spann_args->truth_path);
        if(spann_args->BKTKmeansK > 0) iniReader.SetParameter("SelectHead", "BKTKmeansK", std::to_string(spann_args->BKTKmeansK));
        if(spann_args->Samples > 0) iniReader.SetParameter("SelectHead", "SamplesNumber", std::to_string(spann_args->Samples));
        if(spann_args->TPTNumber > 0) iniReader.SetParameter("BuildHead", "TPTNumber", std::to_string(spann_args->TPTNumber));
        if(spann_args->RefineIterations > 0) iniReader.SetParameter("BuildHead", "RefineIterations", std::to_string(spann_args->RefineIterations));
        if(spann_args->NeighborhoodSize > 0) iniReader.SetParameter("BuildHead", "NeighborhoodSize", std::to_string(spann_args->NeighborhoodSize));
        if(spann_args->CEF > 0) iniReader.SetParameter("BuildHead", "CEF", std::to_string(spann_args->CEF));
        if(spann_args->MaxCheckForRefineGraph > 0) iniReader.SetParameter("BuildHead", "MaxCheckForRefineGraph", std::to_string(spann_args->MaxCheckForRefineGraph));
        if(spann_args->GraphNeighborhoodScale > 0) iniReader.SetParameter("BuildHead", "GraphNeighborhoodScale", std::to_string(spann_args->GraphNeighborhoodScale));
        if(spann_args->NumberOfInitialDynamicPivots > 0) iniReader.SetParameter("BuildHead", "NumberOfInitialDynamicPivots", std::to_string(spann_args->NumberOfInitialDynamicPivots));
        if(spann_args->NumberOfOtherDynamicPivots > 0) iniReader.SetParameter("BuildHead", "NumberOfOtherDynamicPivots", std::to_string(spann_args->NumberOfOtherDynamicPivots));

        for (const auto &section: sections) {
                iniReader.SetParameter(section, "NumberOfThreads", std::to_string(builder_options_->m_threadNum));
            for (const auto &iter: iniReader.GetParameters(section)) {
                std::cout << iter.first << ": " << iter.second << " @ " << section << std::endl;
                if (section == "Base" && iter.first == "indexdirectory") {
                    std::cout << "Skipping [Base][IndexDirectory] => Setting value to specified output folder = '" << builder_options_->m_outputFolder << "'" << std::endl;
                    sptag_vector_index_->SetParameter(iter.first, builder_options_->m_outputFolder, section);
                } else {
                    sptag_vector_index_->SetParameter(iter.first, iter.second, section);
                }
            }
        }

        SPTAG::ErrorCode code;
        std::shared_ptr<SPTAG::VectorSet> vec_set;
        std::shared_ptr<SPTAG::MetadataSet> metadata_set;

        if(n > 0) {
            if (!v) throw std::invalid_argument("Initial data pointer cannot be null when n > 0.");

            uint64_t vec_size_bytes = sizeof(T) * this->dims_ * n;
            SPTAG::ByteArray vec_set_byte_arr = SPTAG::ByteArray::Alloc(vec_size_bytes);
            char *vecBuf = reinterpret_cast<char *>(vec_set_byte_arr.Data());
            memcpy(vecBuf, v, vec_size_bytes);
            vec_set.reset(new SPTAG::BasicVectorSet(vec_set_byte_arr, builder_options_->m_inputValueType, this->dims_, n));
        }
        metadata_set = nullptr;

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
    void SPANNIndex<T>::topk(const idx_t &nq,
                             T *query,
                             idx_t *ids,
                             T *distances,
                             double& peak_wss_mb,
                             const idx_t &k,
                             const DISTANCE_METRIC &distance_metric,
                             const float& c,
                             const NamedArgs* args) const {

        if(!query) throw std::runtime_error("query cannot be nullptr");

        const auto * spann_args = parse_spann_named_args(args);

        std::shared_ptr<SearcherOptions> options(new SearcherOptions);
        options->m_indexFolder = builder_options_->m_outputFolder;
        options->m_dimension = builder_options_->m_dimension;
        options->m_inputValueType = builder_options_->m_inputValueType;
        options->m_inputFileType = SPTAG::VectorFileType::XVEC;
        options->m_K = (int)k;
        options->m_batch = spann_args->batch_size;
        options->m_threadNum = spann_args->thread_num;

        sptag_vector_index_->SetQuantizerADC(options->m_enableADC);

        SPTAG::Helper::IniReader iniReader;
        std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index"};

        if(builder_options_->m_builderConfigFile.empty())
            throw std::runtime_error("no ini file, '" + builder_options_->m_builderConfigFile + "'");

        if (iniReader.LoadIniFile(builder_options_->m_builderConfigFile) != SPTAG::ErrorCode::Success)
            throw std::runtime_error("cannot open index configure file, '" + builder_options_->m_builderConfigFile + "'");

        if constexpr (std::is_same_v<T, int8_t>)
            iniReader.SetParameter("Base", "ValueType", "Int8");
        else if constexpr (std::is_same_v<T, float>)
            iniReader.SetParameter("Base", "ValueType", "Float");

        if(!spann_args->truth_path.empty()) iniReader.SetParameter("Base", "TruthPath", spann_args->truth_path);
        if(spann_args->BKTKmeansK > 0) iniReader.SetParameter("SelectHead", "BKTKmeansK", std::to_string(spann_args->BKTKmeansK));
        if(spann_args->Samples > 0) iniReader.SetParameter("SelectHead", "SamplesNumber", std::to_string(spann_args->Samples));
        if(spann_args->TPTNumber > 0) iniReader.SetParameter("BuildHead", "TPTNumber", std::to_string(spann_args->TPTNumber));
        if(spann_args->RefineIterations > 0) iniReader.SetParameter("BuildHead", "RefineIterations", std::to_string(spann_args->RefineIterations));
        if(spann_args->NeighborhoodSize > 0) iniReader.SetParameter("BuildHead", "NeighborhoodSize", std::to_string(spann_args->NeighborhoodSize));
        if(spann_args->CEF > 0) iniReader.SetParameter("BuildHead", "CEF", std::to_string(spann_args->CEF));
        if(spann_args->MaxCheckForRefineGraph > 0) iniReader.SetParameter("BuildHead", "MaxCheckForRefineGraph", std::to_string(spann_args->MaxCheckForRefineGraph));
        if(spann_args->GraphNeighborhoodScale > 0) iniReader.SetParameter("BuildHead", "GraphNeighborhoodScale", std::to_string(spann_args->GraphNeighborhoodScale));
        if(spann_args->NumberOfInitialDynamicPivots > 0) iniReader.SetParameter("BuildHead", "NumberOfInitialDynamicPivots", std::to_string(spann_args->NumberOfInitialDynamicPivots));
        if(spann_args->NumberOfOtherDynamicPivots > 0) iniReader.SetParameter("BuildHead", "NumberOfOtherDynamicPivots", std::to_string(spann_args->NumberOfOtherDynamicPivots));

        for (const auto &section: sections) {
                iniReader.SetParameter(section, "NumberOfThreads", std::to_string(options->m_threadNum));
            for (const auto &iter: iniReader.GetParameters(section)) {
                std::cout << iter.first << ": " << iter.second << " @ " << section;
                if (section == "Base" && iter.first == "indexdirectory") {
                    std::cout << "Skipping [Base][IndexDirectory] => Setting value to specified output folder = '" << builder_options_->m_outputFolder << "'" << std::endl;
                    sptag_vector_index_->SetParameter(iter.first, builder_options_->m_outputFolder, section);
                } else {
                    sptag_vector_index_->SetParameter(iter.first, iter.second, section);
                }
            }
        }
        sptag_vector_index_->UpdateIndex();

        std::shared_ptr<SPTAG::VectorSet> query_vector_set = nullptr;
        std::shared_ptr<SPTAG::MetadataSet> query_meta_set(nullptr);

        if(options->m_queryFile.empty()) {
            uint64_t query_size_bytes = sizeof(T) * this->dims_ * nq;
            SPTAG::ByteArray query_vector_set_byte_arr = SPTAG::ByteArray::Alloc(query_size_bytes);
            char *vecBuf = reinterpret_cast<char *>(query_vector_set_byte_arr.Data());
            memcpy(vecBuf, query, query_size_bytes);
            query_vector_set.reset(new SPTAG::BasicVectorSet(query_vector_set_byte_arr, builder_options_->m_inputValueType, this->dims_, nq));
        } else {
            auto vector_reader = SPTAG::Helper::VectorSetReader::CreateInstance(options);
            if (SPTAG::ErrorCode::Success != vector_reader->LoadFile(options->m_queryFile)) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                exit(1);
            }
            query_vector_set = vector_reader->GetVectorSet(0, options->m_debugQuery);
            query_meta_set = vector_reader->GetMetadataSet();
        }

        int internalResultNum = options->m_K;
        if (sptag_vector_index_->GetIndexAlgoType() == SPTAG::IndexAlgoType::SPANN) {
            int SPANNInternalResultNum;
            if (SPTAG::Helper::Convert::ConvertStringTo<int>(sptag_vector_index_->GetParameter("SearchInternalResultNum", "BuildSSDIndex").c_str(), SPANNInternalResultNum))
                internalResultNum = max(internalResultNum, SPANNInternalResultNum);
        }

        std::vector<SPTAG::QueryResult> results(options->m_batch, SPTAG::QueryResult(nullptr, internalResultNum, options->m_withMeta != 0));
        int baseSquare = SPTAG::COMMON::Utils::GetBase<T>() * SPTAG::COMMON::Utils::GetBase<T>();

        std::vector<std::string> maxCheck = SPTAG::Helper::StrUtils::SplitString(options->m_maxCheck, "#");

        for (int startQuery = 0; startQuery < query_vector_set->Count(); startQuery += options->m_batch) {
            int numQuerys = min(options->m_batch, query_vector_set->Count() - startQuery);
            for (SPTAG::SizeType i = 0; i < numQuerys; i++) results[i].SetTarget(query_vector_set->GetVector(startQuery + i));

            for (const auto & mc : maxCheck) {
                sptag_vector_index_->SetParameter("MaxCheck", mc.c_str());

                for (SPTAG::SizeType i = 0; i < numQuerys; i++) results[i].Reset();

                omp_set_num_threads((int)options->m_threadNum);
                size_t queriesSent = 0;
                #pragma omp parallel
                {
                    size_t qid;
                    while (true) {
                        #pragma omp atomic capture
                        qid = queriesSent++;
                        if (qid < numQuerys) {
                            sptag_vector_index_->SearchIndex(results[qid]);
                        } else {
                            break;
                        }
                    }
                }

                #ifndef _MSC_VER
                struct rusage rusage;
                getrusage(RUSAGE_SELF, &rusage);
//                double peakWSS = (double)(rusage.ru_maxrss * 1024) / (double)1000000000;
                double peakWSS = (double)(rusage.ru_maxrss) / (double)1024;
                #else
                PROCESS_MEMORY_COUNTERS pmc;
                GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
                unsigned long long peakWSS = pmc.PeakWorkingSetSize / 1000000000;
                #endif
                peak_wss_mb = peakWSS;
                std::cout << "peakWSS = " << peakWSS << " MB" << std::endl;
            }

            for (SPTAG::SizeType i = 0; i < numQuerys; i++) {
                for (int j = 0; j < options->m_K; j++) {
                    if constexpr (std::is_same_v<T, int8_t>)
                        *((int8_t*)distances + startQuery * options->m_K + i * options->m_K + j) = (T)results[i].GetResult(j)->Dist / baseSquare;
                    else if constexpr (std::is_same_v<T, int16_t>)
                        *((int16_t*)distances + startQuery * options->m_K + i * options->m_K + j) = (T)results[i].GetResult(j)->Dist / baseSquare;
                    else if constexpr (std::is_same_v<T, uint8_t>)
                        *((uint8_t*)distances + startQuery * options->m_K + i * options->m_K + j) = (T)results[i].GetResult(j)->Dist / baseSquare;
                    else if constexpr (std::is_same_v<T, float>) {
                        *((float*)distances + startQuery * options->m_K + i * options->m_K + j) = (T) results[i].GetResult(j)->Dist / baseSquare;
                    }
                    else throw std::runtime_error("invalid type T in search process");
                    if (results[i].GetResult(j)->VID < 0) {
                        ids[startQuery * options->m_K + i * options->m_K + j] = -1;
                        continue;
                    }
                    ids[startQuery * options->m_K + i * options->m_K + j] = results[i].GetResult(j)->VID;
                }
            }
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
        return sptag_vector_index_->GetNumSamples();
    }

    template <typename T>
    bool SPANNIndex<T>::built() const {
        return this->built_;
    }

    template class SPANNIndex<int8_t>;
    template class SPANNIndex<int16_t>;
    template class SPANNIndex<uint8_t>;
    template class SPANNIndex<float>;
}