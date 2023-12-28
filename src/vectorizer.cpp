#include "vectorizer.hpp"

#include <utility>
#include "utils.hpp"


namespace mvdb {
//    VectorizerMetadata::VectorizerMetadata(std::string  model, const uint64_t& dimensions):
//    model(std::move(model)), dimensions(dimensions) {}
//
//    void VectorizerMetadata::serialize(std::ostream& out) const {
//        serialize_string(out, model);
//        serialize_numeric<uint64_t>(out, dimensions);
//    }
//
//    void VectorizerMetadata::deserialize(std::istream& in) {
//        model = deserialize_string(in);
//        dimensions = deserialize_numeric<uint64_t>(in);
//    }

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