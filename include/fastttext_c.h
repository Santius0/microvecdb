#ifndef FASTTTEXT_C_H
#define FASTTTEXT_C_H

#ifdef __cplusplus
extern "C" {
#endif

    void *create_fasttext_model();
    void train_supervised(void *model, const char *input, const char *output);
    void predict(void *model, const char *text, char *buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif //FASTTTEXT_C_H
