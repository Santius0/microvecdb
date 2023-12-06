#ifndef METADATA_H
#define METADATA_H

#include <string>
#include <vector>

namespace mvdb {

    class CollectionMetadata {
        std::string name;
        std::string indexFilePath;
        std::string dataDirectoryPath;
        size_t recordCount{};
        size_t indexSize{};
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        void serialize(std::ostream& out) const;
        void deserialize(std::istream& in);
        friend class MetadataManager;
    public:
        CollectionMetadata() = default;
        CollectionMetadata(std::string name, std::string indexFilePath, std::string dataDirectoryPath,
            const size_t& recordCount, const size_t& indexSize);
        ~CollectionMetadata() = default;
    };

    class MetadataManager {
        std::string metadataFilePath;
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        std::vector<CollectionMetadata> collections;

        void updateCreatedTimestamp();
        void updateModifiedTimestamp();
    public:
        MetadataManager() = default;
        explicit MetadataManager(const std::string& metadataFilePath);
        ~MetadataManager();
        void createCollection(const CollectionMetadata& metadata);
        void deleteCollection(const std::string& collectionName);
        void load();
        void save();
    };

}

#endif //METADATA_H
