#include "confluxstorage.h"

//#include <zlib.h>
#include <iostream>
//
//std::string compressData(const std::string& data) {
//    uLongf compressedSize = compressBound(data.size());
//    std::vector<char> compressedData(compressedSize);
//
//    if (compress(reinterpret_cast<Bytef*>(&compressedData[0]), &compressedSize,
//                 reinterpret_cast<const Bytef*>(data.data()), data.size()) != Z_OK) {
//        std::cerr << "Compression failed!" << std::endl;
//        return "";
//    }
//
//    return std::string(compressedData.begin(), compressedData.begin() + compressedSize);
//}
//
//std::string decompressData(const std::string& compressedData) {
//    uLongf decompressedSize = compressedData.size() * 10; // Estimate size; adjust as needed
//    std::vector<char> decompressedData(decompressedSize);
//
//    if (uncompress(reinterpret_cast<Bytef*>(&decompressedData[0]), &decompressedSize,
//                   reinterpret_cast<const Bytef*>(compressedData.data()), compressedData.size()) != Z_OK) {
//        std::cerr << "Decompression failed!" << std::endl;
//        return "";
//    }
//
//    return std::string(decompressedData.begin(), decompressedData.begin() + decompressedSize);
//}


ConfluxStorage* ConfluxStorage::instance = nullptr;

ConfluxStorage::ConfluxStorage(const std::string &db_path){
    this->options.create_if_missing = true;
    this->status = rocksdb::DB::Open(this->options, db_path, &db);
    std::cout << this->db->GetLatestSequenceNumber() << std::endl;
}

ConfluxStorage::~ConfluxStorage(){
    if(this->db != nullptr) {
        delete this->db;
        this->db = nullptr;
    }
}

ConfluxStorage* ConfluxStorage::getInstance(const std::string &db_path){
    if (ConfluxStorage::instance == nullptr)
        ConfluxStorage::instance = new ConfluxStorage(db_path);
    return ConfluxStorage::instance;
}

void ConfluxStorage::destroyInstance() {
    if(ConfluxStorage::instance != nullptr) {
        delete ConfluxStorage::instance;
        ConfluxStorage::instance = nullptr;
    }
}

rocksdb::Status ConfluxStorage::getStatus(){
    return this->status;
}

rocksdb::Options ConfluxStorage::getOptions(){
    return this->options;
}

bool ConfluxStorage::put(const std::string& key, const std::string& put_value) {
    this->status = db->Put(rocksdb::WriteOptions(), key, put_value);
    return this->status.ok();
}

bool ConfluxStorage::get(const std::string& key, std::string &fetched_value){
    this->status = db->Get(rocksdb::ReadOptions(), key, &fetched_value);
    return this->status.ok();
}

unsigned long ConfluxStorage::get_latest_sequence_number() {
    return this->db->GetLatestSequenceNumber();
}