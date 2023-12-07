#ifndef METADATA_H
#define METADATA_H

#include "vector_collection.hpp"
#include "serializable.h"
#include <ostream>
#include <string>
#include <vector>

namespace mvdb {

    class Metadata final: public Serializable {
        std::string metadataFilePath;
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        std::vector<CollectionMetadata> collections_;
        void updateCreatedTimestamp();
        void updateModifiedTimestamp();
    protected:
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
    public:
        Metadata() = default;
        explicit Metadata(const std::string& metadataFilePath);
        ~Metadata() override;
        void addCollection(const CollectionMetadata& metadata);
        void deleteCollection(const std::string& collectionName);
        void load();
        void save();
        friend std::ostream& operator<<(std::ostream& os, const Metadata& obj) {
            return os
                   << "metadataFilePath: " << obj.metadataFilePath << std::endl
                   << " createdTimestamp: " << obj.createdTimestamp << std::endl
                   << " modifiedTimestamp: " << obj.modifiedTimestamp << std::endl;
            // << " collections: " << obj.collections;
        }

        friend std::ostream& operator<<(std::ostream& os, const Metadata* obj) {
            return os << "*(" << *obj << ")";
        }
    };

}

#endif //METADATA_H
