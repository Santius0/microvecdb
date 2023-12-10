#ifndef VECTORIZER_H
#define VECTORIZER_H

#include "constants.hpp"
#include "serializable.hpp"
#include <fasttext.h>
#include <memory>


namespace mvdb {

    class VectorizerMetadata : public Serializable{
        std::string model;
        uint64_t dimensions{};
    protected:
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        friend class Vectorizer;
        friend class VectorCollectionMetadata;
        friend class Metadata;
    public:
        VectorizerMetadata() = default;
        explicit VectorizerMetadata(std::string  model, const uint64_t& dimensions);
        ~VectorizerMetadata() override = default;
    };

    class Vectorizer {
        std::unique_ptr<fasttext::FastText> model;
        bool loaded = false;
    public:
        explicit Vectorizer(const VectorizerMetadata& metadata);
        ~Vectorizer() = default;
         fasttext::Vector get_word_vector(const std::string& word) const;
        // void train_supervised(const char *input, const char *output);
        // void predict(const char *text, char *buffer, int buffer_size);
    };

}


#endif //VECTORIZER_H
