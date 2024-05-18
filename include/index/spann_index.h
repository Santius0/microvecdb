#ifndef MICROVECDB_SPANN_INDEX_H
#define MICROVECDB_SPANN_INDEX_H

#include "index.h"

#include <memory>
#include <thread>
#include <iostream>

#include <inc/Helper/VectorSetReader.h>
#include <inc/Core/VectorIndex.h>
#include <inc/Core/Common.h>
#include <inc/Helper/SimpleIniReader.h>

#include <Core/Common/DistanceUtils.h>

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/AsyncFileReader.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/VectorIndex.h"
#include <algorithm>
#include <iomanip>
#include <set>
#include <atomic>
#include <ctime>
#include <thread>
#include <chrono>

#include <inc/Core/SPANN/Index.h>

#include <type_traits>

#define max_threads std::thread::hardware_concurrency();

namespace mvdb::index {

    static const unsigned int hw_concurrency = std::thread::hardware_concurrency();

    class BuilderOptions : public SPTAG::Helper::ReaderOptions {
    public:
        BuilderOptions() : SPTAG::Helper::ReaderOptions(SPTAG::VectorValueType::Float, 0,
                                                        SPTAG::VectorFileType::TXT, "|",
                                                        hw_concurrency) {
            AddRequiredOption(m_outputFolder, "-o", "--outputfolder", "Output folder.");
            AddRequiredOption(m_indexAlgoType, "-a", "--algo", "Index Algorithm type.");
            AddOptionalOption(m_inputFiles, "-i", "--input", "Input raw data.");
            AddOptionalOption(m_builderConfigFile, "-c", "--config", "Config file for builder.");
            AddOptionalOption(m_quantizerFile, "-pq", "--quantizer", "Quantizer File");
            AddOptionalOption(m_metaMapping, "-m", "--metaindex", "Enable delete vectors through metadata");
        }

        ~BuilderOptions() {}

        std::string m_inputFiles;

        std::string m_outputFolder;

        SPTAG::IndexAlgoType m_indexAlgoType;

        std::string m_builderConfigFile;

        std::string m_quantizerFile;

        bool m_metaMapping = false;
    };

    class SearcherOptions : public SPTAG::Helper::ReaderOptions {
    public:
        SearcherOptions() : SPTAG::Helper::ReaderOptions(SPTAG::VectorValueType::Float, 0,
                                                         SPTAG::VectorFileType::TXT, "|",
                                                         std::thread::hardware_concurrency())
        {
            AddRequiredOption(m_queryFile, "-i", "--input", "Input raw data.");
            AddRequiredOption(m_indexFolder, "-x", "--index", "Index folder.");
            AddOptionalOption(m_truthFile, "-r", "--truth", "Truth file.");
            AddOptionalOption(m_resultFile, "-o", "--result", "Output result file.");
            AddOptionalOption(m_maxCheck, "-m", "--maxcheck", "MaxCheck for index.");
            AddOptionalOption(m_withMeta, "-a", "--withmeta", "Output metadata instead of vector id.");
            AddOptionalOption(m_K, "-k", "--KNN", "K nearest neighbors for search.");
            AddOptionalOption(m_truthK, "-tk", "--truthKNN", "truth set number.");
            AddOptionalOption(m_dataFile, "-df", "--data", "original data file.");
            AddOptionalOption(m_dataFileType, "-dft", "--dataFileType", "original data file type. (TXT, or DEFAULT)");
            AddOptionalOption(m_batch, "-b", "--batchsize", "Batch query size.");
            AddOptionalOption(m_genTruth, "-g", "--gentruth", "Generate truth file.");
            AddOptionalOption(m_debugQuery, "-q", "--debugquery", "Debug query number.");
            AddOptionalOption(m_enableADC, "-adc", "--adc", "Enable ADC Distance computation");
            AddOptionalOption(m_outputformat, "-of", "--ouputformat", "0: TXT 1: BINARY.");
        }

        ~SearcherOptions() = default;

        std::string m_queryFile;

        std::string m_indexFolder;

        std::string m_dataFile = "";

        std::string m_truthFile = "";

        std::string m_resultFile = "";

        std::string m_maxCheck = "8192";

        SPTAG::VectorFileType m_dataFileType = SPTAG::VectorFileType::DEFAULT;

        int m_withMeta = 0;

        int m_K = 32;

        int m_truthK = -1;

        int m_batch = 10000;

        int m_genTruth = 0;

        int m_debugQuery = -1;

        bool m_enableADC = false;

        int m_outputformat = 0;
    };

    template <typename T>
    int Process(const std::shared_ptr<SearcherOptions>& options, SPTAG::VectorIndex& index,
                idx_t *ids = nullptr, void *distances = nullptr,
                std::shared_ptr<SPTAG::VectorSet> queryVectors = nullptr,
                std::shared_ptr<SPTAG::MetadataSet> queryMetas = nullptr) {
        std::ofstream log("Recall-result.out", std::ios::app);
        if (!log.is_open()) {
            SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "ERROR: Cannot open logging file!\n");
            exit(-1);
        }

        if(queryVectors == nullptr) {
            auto vectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(options);
            if (SPTAG::ErrorCode::Success != vectorReader->LoadFile(options->m_queryFile)) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                exit(1);
            }
            queryVectors = vectorReader->GetVectorSet(0, options->m_debugQuery);
            queryMetas = vectorReader->GetMetadataSet();
            std::cout << "USING FVECS FILE FOR QUERY!!\n";
        } else {
            std::cout << "USING MEMORY FOR QUERY!!\n";
        }

        std::shared_ptr<SPTAG::Helper::ReaderOptions> dataOptions(new SPTAG::Helper::ReaderOptions(queryVectors->GetValueType(), queryVectors->Dimension(), options->m_dataFileType));
        auto dataReader = SPTAG::Helper::VectorSetReader::CreateInstance(dataOptions);
        std::shared_ptr<SPTAG::VectorSet> dataVectors;
        if (!options->m_dataFile.empty()) {
            if (SPTAG::ErrorCode::Success != dataReader->LoadFile(options->m_dataFile)) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Failed to read data file.\n");
                exit(1);
            }
            dataVectors = dataReader->GetVectorSet();
        }

        std::shared_ptr<SPTAG::Helper::DiskIO> ftruth;
        int truthDim = 0;
        if (!options->m_truthFile.empty()) {
            if (options->m_genTruth) {
                if (dataVectors == nullptr) {
                    SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot load data vectors to generate groundtruth! Please speicify data vector file by setting -df option.\n");
                    exit(1);
                }
                SPTAG::COMMON::TruthSet::GenerateTruth<T>(queryVectors, dataVectors, options->m_truthFile, index.GetDistCalcMethod(), options->m_truthK,
                                                   (options->m_truthFile.find("bin") != std::string::npos) ? SPTAG::TruthFileType::DEFAULT : SPTAG::TruthFileType::TXT, index.m_pQuantizer);
            }

            ftruth = SPTAG::f_createIO();
            if (ftruth == nullptr || !ftruth->Initialize(options->m_truthFile.c_str(), std::ios::in | std::ios::binary)) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for read!\n", options->m_truthFile.c_str());
                exit(1);
            }
            if (options->m_truthFile.find("bin") != std::string::npos) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Load binary truth...\n");
            }
            else {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Load txt truth...\n");
            }
        }

        std::shared_ptr<SPTAG::Helper::DiskIO> fp;
        if (!options->m_resultFile.empty()) {
            fp = SPTAG::f_createIO();
            if (fp == nullptr || !fp->Initialize(options->m_resultFile.c_str(), std::ios::out | std::ios::binary)) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for write!\n", options->m_resultFile.c_str());
            }

            if (options->m_outputformat == 1) {
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Using output format binary...");

                int32_t i32Val = queryVectors->Count();
                if (fp->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                    exit(1);
                }
                i32Val = options->m_K;
                if (fp->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                    exit(1);
                }
            }
        }

        std::vector<std::string> maxCheck = SPTAG::Helper::StrUtils::SplitString(options->m_maxCheck, "#");
        if (options->m_truthK < 0) options->m_truthK = options->m_K;

        std::vector<std::set<SPTAG::SizeType>> truth(options->m_batch);
        int internalResultNum = options->m_K;
        if (index.GetIndexAlgoType() == SPTAG::IndexAlgoType::SPANN) {
            int SPANNInternalResultNum;
            if (SPTAG::Helper::Convert::ConvertStringTo<int>(index.GetParameter("SearchInternalResultNum", "BuildSSDIndex").c_str(), SPANNInternalResultNum))
                internalResultNum = max(internalResultNum, SPANNInternalResultNum);
        }
        std::vector<SPTAG::QueryResult> results(options->m_batch, SPTAG::QueryResult(NULL, internalResultNum, options->m_withMeta != 0));
        std::vector<float> latencies(options->m_batch, 0);
        int baseSquare = SPTAG::COMMON::Utils::GetBase<T>() * SPTAG::COMMON::Utils::GetBase<T>();

        SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "[query]\t\t[maxcheck]\t[avg] \t[99%] \t[95%] \t[recall] \t[qps] \t[mem]\n");

        std::vector<float> totalAvg(maxCheck.size(), 0.0), total99(maxCheck.size(), 0.0), total95(maxCheck.size(), 0.0), totalRecall(maxCheck.size(), 0.0), totalLatency(maxCheck.size(), 0.0);

        for (int startQuery = 0; startQuery < queryVectors->Count(); startQuery += options->m_batch) {
            int numQuerys = min(options->m_batch, queryVectors->Count() - startQuery);
            for (SPTAG::SizeType i = 0; i < numQuerys; i++) results[i].SetTarget(queryVectors->GetVector(startQuery + i));
            if (ftruth != nullptr) SPTAG::COMMON::TruthSet::LoadTruth(ftruth, truth, numQuerys, truthDim, options->m_truthK, (options->m_truthFile.find("bin") != std::string::npos)? SPTAG::TruthFileType::DEFAULT : SPTAG::TruthFileType::TXT);

            for (int mc = 0; mc < maxCheck.size(); mc++) {
                index.SetParameter("MaxCheck", maxCheck[mc].c_str());

                for (SPTAG::SizeType i = 0; i < numQuerys; i++) results[i].Reset();

                std::atomic_size_t queriesSent(0);
                std::vector<std::thread> threads;
                threads.reserve(options->m_threadNum);
                auto batchstart = std::chrono::high_resolution_clock::now();

                for (std::uint32_t i = 0; i < options->m_threadNum; i++) {
                    threads.emplace_back([&, i] {
                        SPTAG::NumaStrategy ns = (index.GetIndexAlgoType() == SPTAG::IndexAlgoType::SPANN)? SPTAG::NumaStrategy::SCATTER: SPTAG::NumaStrategy::LOCAL; // Only for SPANN, we need to avoid IO threads overlap with search threads.
                        SPTAG::Helper::SetThreadAffinity(i, threads[i], ns, SPTAG::OrderStrategy::ASC);

                        size_t qid = 0;
                        while (true) {
                            qid = queriesSent.fetch_add(1);
                            if (qid < numQuerys) {
                                auto t1 = std::chrono::high_resolution_clock::now();
                                index.SearchIndex(results[qid]);
                                auto t2 = std::chrono::high_resolution_clock::now();
                                latencies[qid] = (float)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0);
                            }
                            else {
                                return;
                            }
                        }
                    });
                }
                for (auto& thread : threads) { thread.join(); }

                auto batchend = std::chrono::high_resolution_clock::now();
                float batchLatency = (float)(std::chrono::duration_cast<std::chrono::microseconds>(batchend - batchstart).count() / 1000000.0);

                float timeMean = 0, timeMin = SPTAG::MaxDist, timeMax = 0, timeStd = 0;
                for (int qid = 0; qid < numQuerys; qid++)
                {
                    timeMean += latencies[qid];
                    if (latencies[qid] > timeMax) timeMax = latencies[qid];
                    if (latencies[qid] < timeMin) timeMin = latencies[qid];
                }
                timeMean /= numQuerys;
                for (int qid = 0; qid < numQuerys; qid++) timeStd += ((float)latencies[qid] - timeMean) * ((float)latencies[qid] - timeMean);
                timeStd = std::sqrt(timeStd / numQuerys);
                log << timeMean << " " << timeStd << " " << timeMin << " " << timeMax << " ";

                std::sort(latencies.begin(), latencies.begin() + numQuerys);
                float l99 = latencies[SPTAG::SizeType(numQuerys * 0.99)];
                float l95 = latencies[SPTAG::SizeType(numQuerys * 0.95)];

                float recall = 0;
                if (ftruth != nullptr) {
                    recall = SPTAG::COMMON::TruthSet::CalculateRecall<T>(&index, results, truth, options->m_K, options->m_truthK, queryVectors, dataVectors, numQuerys, &log, options->m_debugQuery > 0);
                }

#ifndef _MSC_VER
                struct rusage rusage;
                getrusage(RUSAGE_SELF, &rusage);
                unsigned long long peakWSS = rusage.ru_maxrss * 1024 / 1000000000;
#else
                PROCESS_MEMORY_COUNTERS pmc;
            GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
            unsigned long long peakWSS = pmc.PeakWorkingSetSize / 1000000000;
#endif
                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "%d-%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t\t%.4f\t\t%lluGB\n", startQuery, (startQuery + numQuerys), maxCheck[mc].c_str(), timeMean, l99, l95, recall, (numQuerys / batchLatency), peakWSS);
                totalAvg[mc] += timeMean * numQuerys;
                total95[mc] += l95 * numQuerys;
                total99[mc] += l99 * numQuerys;
                totalRecall[mc] += recall * numQuerys;
                totalLatency[mc] += batchLatency;
            }

            if (fp != nullptr) {
                if (options->m_outputformat == 0) {
                    for (SPTAG::SizeType i = 0; i < numQuerys; i++)
                    {
                        if (queryMetas != nullptr) {
                            SPTAG::ByteArray qmeta = queryMetas->GetMetadata(startQuery + i);
                            if (fp->WriteBinary(qmeta.Length(), (const char*)qmeta.Data()) != qmeta.Length()) {
                                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot write qmeta %d bytes!\n", qmeta.Length());
                                exit(1);
                            }
                        }
                        else {
                            std::string qid = std::to_string(startQuery + i);
                            if (fp->WriteBinary(qid.length(), qid.c_str()) != qid.length()) {
                                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot write qid %d bytes!\n", qid.length());
                                exit(1);
                            }
                        }
                        fp->WriteString(":");
                        for (int j = 0; j < options->m_K; j++)
                        {
                            std::string sd = std::to_string(results[i].GetResult(j)->Dist / baseSquare);
                            if (fp->WriteBinary(sd.length(), sd.c_str()) != sd.length()) {
                                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot write dist %d bytes!\n", sd.length());
                                exit(1);
                            }
                            fp->WriteString("@");
                            if (results[i].GetResult(j)->VID < 0) {
                                fp->WriteString("NULL|");
                                continue;
                            }

                            if (!options->m_withMeta) {
                                std::string vid = std::to_string(results[i].GetResult(j)->VID);
                                if (fp->WriteBinary(vid.length(), vid.c_str()) != vid.length()) {
                                    SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot write vid %d bytes!\n", sd.length());
                                    exit(1);
                                }
                            }
                            else {
                                SPTAG::ByteArray vm = index.GetMetadata(results[i].GetResult(j)->VID);
                                if (fp->WriteBinary(vm.Length(), (const char*)vm.Data()) != vm.Length()) {
                                    SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot write vmeta %d bytes!\n", vm.Length());
                                    exit(1);
                                }
                            }
                            fp->WriteString("|");
                        }
                        fp->WriteString("\n");
                    }
                }
                else {
                    for (SPTAG::SizeType i = 0; i < numQuerys; ++i)
                    {
                        for (int j = 0; j < options->m_K; ++j)
                        {
                            SPTAG::SizeType i32Val = results[i].GetResult(j)->VID;
                            if (fp->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                exit(1);
                            }

                            float fVal = results[i].GetResult(j)->Dist;
                            if (fp->WriteBinary(sizeof(fVal), reinterpret_cast<char*>(&fVal)) != sizeof(fVal)) {
                                SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                exit(1);
                            }
                        }
                    }
                }
            } else {
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
        for (int mc = 0; mc < maxCheck.size(); mc++)
            SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "%d-%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", 0, queryVectors->Count(), maxCheck[mc].c_str(), (totalAvg[mc] / queryVectors->Count()), (total99[mc] / queryVectors->Count()), (total95[mc] / queryVectors->Count()), (totalRecall[mc] / queryVectors->Count()), (queryVectors->Count() / totalLatency[mc]));
        SPTAG::SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Output results finish!\n");
        if (fp != nullptr) fp->ShutDown();
        log.close();
        return 0;
    }

    struct SPANNIndexNamedArgs final : NamedArgs {
            std::string build_config_path;
            std::string quantizer_path;
            bool meta_mapping = false;
            bool normalized = false;
            uint32_t thread_num = hw_concurrency;
            SPANNIndexNamedArgs() = default;
            ~SPANNIndexNamedArgs() override = default;
    };

    template <typename T = float>
    class SPANNIndex final : public Index<T> {
        std::shared_ptr<SPTAG::VectorIndex> sptag_vector_index_;
        std::shared_ptr<BuilderOptions> builder_options_;
        friend std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>* obj);
    protected:
        void save_(const std::string& path) const override;
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        SPANNIndex();
        ~SPANNIndex() override = default;
        SPANNIndex(const SPANNIndex&) = delete;
        SPANNIndex& operator=(const SPANNIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) override;
        void open(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void topk(const idx_t& nq, T* query, const std::string& query_path,
                  const std::string& result_path, idx_t* ids, T* distances, const idx_t& k,
                  const DISTANCE_METRIC& distance_metric, const float& c) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] bool built() const override;
    };

    extern template class SPANNIndex<int8_t>;
    extern template class SPANNIndex<int16_t>;
    extern template class SPANNIndex<uint8_t>;
    extern template class SPANNIndex<float>;
}

#endif //MICROVECDB_SPANN_INDEX_H
