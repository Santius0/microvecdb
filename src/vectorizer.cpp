#include "vectorizer.h"
#include "utils.h"


namespace mvdb {
    Vectorizer::Vectorizer(const std::string& model_path, const uint64_t& dimensions) {
        auto *raw_ptr = new fasttext::FastText();
        raw_ptr->loadModel(model_path);     // TODO: make model load faster
        loaded = true;
        model.reset(raw_ptr);
    }

    fasttext::Vector Vectorizer::get_word_vector(const std::string& word) const {
        if(!loaded) throw std::runtime_error("vectorizer does not have any model loaded");
        auto word_vec = fasttext::Vector(model->getDimension());
        model->getWordVector(word_vec, word);
        return word_vec;
    }

    // void Vectorizer::train_supervised(const char *input, const char *output) {
    //     // static_cast<fasttext::FastText*>(model)->train("supervised", input, output);
    // }
    //
    // void Vectorizer::predict(const char *text, char *buffer, int buffer_size) {
    //     // Implement prediction logic
    //     // Use the FastText model to predict and store the result in `buffer`
    // }

}