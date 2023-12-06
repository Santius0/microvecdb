#ifndef METADATA_H
#define METADATA_H

#include "vector_collection.hpp"
#include <ostream>
#include <string>
#include <vector>

namespace mvdb {

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
        void addCollection(const CollectionMetadata& metadata);
        void deleteCollection(const std::string& collectionName);
        void load();
        void save();
        friend std::ostream& operator<<(std::ostream& os, const MetadataManager& obj) {
            return os
                   << "metadataFilePath: " << obj.metadataFilePath << std::endl
                   << " createdTimestamp: " << obj.createdTimestamp << std::endl
                   << " modifiedTimestamp: " << obj.modifiedTimestamp << std::endl;
            // << " collections: " << obj.collections;
        }

        friend std::ostream& operator<<(std::ostream& os, const MetadataManager* obj) {
            return os << "*(" << *obj << ")";
        }
    };

}

#endif //METADATA_H
