#include "vectorizer.h"


namespace mvdb {

    Vectorizer::Vectorizer(const std::string& model_path, const int& dims): model_path(model_path), dims(dims) {
        auto *raw_ptr = new fasttext::FastText();
        raw_ptr->loadModel(model_path);
        model.reset(raw_ptr);
    }

    fasttext::Vector Vectorizer::get_word_vector(const std::string& word) {
        auto word_vec = fasttext::Vector(model->getDimension());
        model->getWordVector(word_vec, word);
    }

}