#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <omp.h>

#include "kv_store.h"
#include "vector_index.h"

void fetch_and_print(const kv_store_t* kv_store, const char* key) {
     char* value = kv_store_get(kv_store, key);
     printf("%s => %s", key, value);
     if(value) free(value);
}

int main() {
    // #pragma omp parallel
    // {
        // printf("Thread %d\n", omp_get_thread_num());
    // }

    kv_store_t* kv_store = kv_store_init("./testdb");

    kv_store_put(kv_store, "key1", "key value on 24/11/23 at 10:20AM");
    kv_store_put(kv_store, "key10", "key value on 24/11/23 at 10:20AM");
    char *key = malloc(sizeof(char)*21);
    kv_store_put_auto(kv_store, "auto_inserted value", key);
    kv_store_put_auto(kv_store, "auto_inserted value", key);
    kv_store_put_auto(kv_store, "auto_inserted value", key);
    kv_store_put_auto(kv_store, "auto_inserted value", key);
    kv_store_put_auto(kv_store, "auto_inserted value", key);

    if(strlen(key) > 0) {
        fetch_and_print(kv_store, key);
    } else {
        printf("failure\n");
    }

    free(key);

    kv_store_free(kv_store);

    vector_index_test();
    return 0;
}
