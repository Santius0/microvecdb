#ifndef VECTORIZER_H
#define VECTORIZER_H

#include <fasttext.h>
#include <memory>


namespace mvdb {

    enum VectorizerModelType {
        MULTI_LANG = 0
    };

    class Vectorizer {
        std::unique_ptr<fasttext::FastText> model;
        std::string model_path;
        int dims;
    public:
        explicit Vectorizer(const std::string& model_path, const int& dims);
        ~Vectorizer() = default;
        [[nodiscard]] fasttext::Vector get_word_vector(const std::string& word) const;
        void train_supervised(const char *input, const char *output);
        void predict(const char *text, char *buffer, int buffer_size);
    };

}


#endif //VECTORIZER_H
