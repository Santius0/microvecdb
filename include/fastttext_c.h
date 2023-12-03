#ifndef FASTTTEXT_C_H
#define FASTTTEXT_C_H

#ifdef __cplusplus
extern "C" {
#endif

    void *fasttext_model_init();
    void* fasttext_model_load(const char* path);
    char* fasttext_model_get_word_vec(const char* word, void* model);
    // void train_supervised(void *model, const char *input, const char *output);
    // void predict(void *model, const char *text, char *buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif //FASTTTEXT_C_H
