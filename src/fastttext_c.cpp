#include "fastttext_c.h"
#include <fasttext.h>

#include <iostream>

void* fasttext_model_init() {
    return new fasttext::FastText();
}

void* fasttext_model_load(const char* path) {
    auto* model = new fasttext::FastText();
    model->loadModel(path);
    return model;
}

char* fasttext_model_get_word_vec(const char* word, void* model) {
    const auto* fasttext_model = static_cast<fasttext::FastText*>(model);
    auto* word_vec = new fasttext::Vector(fasttext_model->getDimension());
    std::cout << word_vec << std::endl;
    fasttext_model->getWordVector(*word_vec, word);
    return (char*)word_vec;
}

// void train_supervised(void *model, const char *input, const char *output) {
//     // static_cast<fasttext::FastText*>(model)->train("supervised", input, output);
// }
//
// void predict(void *model, const char *text, char *buffer, int buffer_size) {
//     // Implement prediction logic
//     // Use the FastText model to predict and store the result in `buffer`
// }
