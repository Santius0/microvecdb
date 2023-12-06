#ifndef METADATA_H
#define METADATA_H

#include <string>
#include <vector>

namespace mvdb {

    class CollectionMetadata {
        std::string name;
        std::string indexFilePath;
        std::string dataDirectoryPath;
        // std::map<std::string, std::string> schema; // Key: field name, Value: field type
        size_t recordCount;
        size_t indexSize;
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        friend class MetadataManager;
    public:
        CollectionMetadata(std::string name, std::string indexFilePath, std::string dataDirectoryPath,
            std::string recordCount, size_t indexSize, std::string createdTimestamp, std::string modifiedTimestamp);
        ~CollectionMetadata() = default;
    };

    class MetadataManager {
        std::string metadataFilePath;
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        std::vector<CollectionMetadata> collections;
    public:
        explicit MetadataManager(const std::string& metadataFilePath);
        ~MetadataManager();
        void createCollection(const CollectionMetadata& metadata);
        void deleteCollection(const std::string& collectionName);
        void load();
        void save();
    };

}

#endif //METADATA_H
