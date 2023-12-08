#include "micrvecdb.hpp"
#include "utils.hpp"
#include "constants.hpp"
#include <filesystem>
#include <stdexcept>

namespace mvdb {

    MicroVecDB::MicroVecDB(const std::string& path, const std::string& dbname): path(trim(path)), dbname(dbname) {
        if(!std::filesystem::exists(path))  // if directory doesn't exist create it
            std::filesystem::create_directory(path);
        else if(!std::filesystem::is_directory(path)) // else if file exists but is not a directory throw error
            throw::std::runtime_error("invalid database path \"" + path + "\"");

        metadata_ = std::make_unique<Metadata>(this->path
            + std::filesystem::path::preferred_separator + dbname + META_FILE_EXTENSION);

        //don't need this anymore. makes more sense to just load the metadata file and only load from the collections
        //when neccessary
        // for (const auto& entry : std::filesystem::directory_iterator(path)) {
            // if (std::filesystem::is_directory(entry.path())) {
                // this->collections_.push_back(std::unique_ptr<VectorCollection>(VectorCollection::load(entry.path())));
                // this->collection_paths_.push_back(entry.path());
            // }
        // }
    }

    void MicroVecDB::create_collection(const std::string& name, const uint64_t& dimensions, const std::string& model) {
        const std::string collectionFilePath = path + std::filesystem::path::preferred_separator + name;
        const std::string indexFilePath = collectionFilePath + std::filesystem::path::preferred_separator + name + INDEX_EXT;
        const std::string dataDirectoryPath = collectionFilePath + std::filesystem::path::preferred_separator + name + KV_STORE_EXT;

        auto db_options = rocksdb::Options();
        db_options.create_if_missing = true; // TODO: have user enter this
        const KvStoreMetadata kv_store_metadata = KvStoreMetadata(dataDirectoryPath, db_options);

        // TODO: fix for multiple types
        const VectorIndexMetadata vector_index_metadata = VectorIndexMetadata(indexFilePath, dimensions, VectorIndexType::FLAT);

        const VectorizerMetadata vectorizer_metadata = VectorizerMetadata(model, dimensions);

        const CollectionMetadata collection_metadata = CollectionMetadata(name, collectionFilePath, 0,
            kv_store_metadata, vector_index_metadata, vectorizer_metadata);

        auto* vc = new VectorCollection(collection_metadata);

        collections_.emplace_back(vc); // TODO: revisit

        metadata_->addCollection(collection_metadata);

        metadata_->save();
    }

    VectorCollection* MicroVecDB::collection(const std::string& name) const {
        for(size_t i = 0; i < metadata_->collections_.size(); i++) {
            if(name == metadata_->collections_[i].name) return collections_[i].get();
        }
        throw std::out_of_range("no collection \"" + name + "\" found\n");
        return nullptr;
    }

}