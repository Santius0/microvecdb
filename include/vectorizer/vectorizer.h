#ifndef VECTORIZER_H
#define VECTORIZER_H

#include "constants.h"
#include "serializable.h"
#include <fasttext.h>
#include <memory>


namespace mvdb {

    enum class VectorizerModelType {
        FASTTEXT = 0
    };

    class Vectorizer {
        std::unique_ptr<fasttext::FastText> model;
        bool loaded = false;
    public:
        explicit Vectorizer(const std::string& model_path, const uint64_t& dimensions);
        ~Vectorizer() = default;
         fasttext::Vector get_word_vector(const std::string& word) const;
        // void train_supervised(const char *input, const char *output);
        // void predict(const char *text, char *buffer, int buffer_size);
    };

}


#endif //VECTORIZER_H
