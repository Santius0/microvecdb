#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <omp.h>
#include <time.h>

#include "kv_store.h"
#include "vector_index.h"

#include "tests.h"

void fetch_and_print(const kv_store_t* kv_store, const char* key) {
    if(strlen(key) > 0) {
        char* value = kv_store_get(kv_store, key);
        printf("%s => %s\n", key, value);
        if(value) free(value);
    } else {
        printf("strlen(key) is 0\n");
    }
}

// Function to generate random float data
void generate_random_data(float* data, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
    }
}

bool verify_kv_store(kv_store_t* store, kv_pair_t* pairs, size_t num_pairs) {
    for (size_t i = 0; i < num_pairs; ++i) {
        char* value = kv_store_get(store, pairs[i].key);
        printf("found: %s => %s      .vs.     %s\n\n", pairs[i].key, value, pairs[i].value);
        if (!value || strcmp(value, pairs[i].value) != 0) {
            if (value) free(value); // Assuming kv_store_get allocates memory for value
            return false;
        }
        free(value); // Assuming kv_store_get allocates memory for value
    }
    return true;
}

int test_storage() {
    kv_store_t* store = kv_store_init("./testdb", false, false);

    if (!store) {
        fprintf(stderr, "Failed to initialize key-value store\n");
        return 1;
    }

    // kv_pair_t pairs[] = {
    //     {"key1", "value11"},
    //     {"key2", "value22"},
    //     {"key3", "value33"}
    // };
    kv_pair_t pairs[] = {
        {NULL, "value11"},
        {NULL, "value22"},
        {NULL, "value33"}
    };
    size_t num_pairs = sizeof(pairs) / sizeof(pairs[0]);

    if (!kv_store_put_many(store, pairs, num_pairs, true)) {
        fprintf(stderr, "Failed to write key-value pairs to the store\n");
        kv_store_free(store);
        return 1;
    }

    // Verify the data was written correctly
    if (verify_kv_store(store, pairs, num_pairs)) {
        printf("All key-value pairs were successfully verified.\n");
    } else {
        fprintf(stderr, "Verification failed for key-value pairs.\n");
        kv_store_free(store);
        return 1;
    }

    // kv_store_free(store);
    
    if(store) {

        // char *test_file_key = "test_file_key";
        // const bool insert_success = kv_store_put_file_name(kv_store, test_file_key, "./eye_pic.png", false);
        // if(insert_success)
        // printf("SUCCESS\n");
        // else
        // printf("FAILURE\n");
        //
        // char *filepath_or_buffer = "./eye_pic_fetched.png";
        // const bool retrieve_success = kv_store_get_file_name(kv_store, test_file_key, filepath_or_buffer, true);
        // if(retrieve_success)
        //     printf("SUCCESS\n");
        // else
        //     printf("FAILURE\n");

        kv_pair_t pair;
        pair.key = "keys_for_days";
        pair.value = "key value on 24/11/23 at 10:20AM";
        kv_store_put(store, pair, false);
        fetch_and_print(store, "keys_for_days");

        pair.key = "keys_for_days_2";
        pair.value = "key value on 24/11/23 at 10:20AM22222222222222";
        kv_store_put(store, pair, false);
        fetch_and_print(store, "keys_for_days_2");

        // char *key = malloc(sizeof(char)*21);
        // kv_store_put(store, key, "auto_inserted value", true);
        // fetch_and_print(store, key);
        // kv_store_put(store, key, "auto_inserted value", true);
        // fetch_and_print(store, key);
        // kv_store_put(store, key, "auto_inserted value", true);
        // fetch_and_print(store, key);
        // kv_store_put(store, key, "auto_inserted value", true);
        // fetch_and_print(store, key);
        // kv_store_put(store, key, "auto_inserted value", true);
        // fetch_and_print(store, key);
        // if(key) free(key);

        kv_store_free(store);
    }
    return 0;
}

int main() {
    vector_index *idx = vector_index_load("test.index");
    // test_storage();
    // #pragma omp parallel
    // {
        // printf("Thread %d\n", omp_get_thread_num());
    // }

    const char *csv_file_path = "./test_data.csv", *index_file_path = "./test.index";
    // generate_and_save_vectors(file_path);

    // create_and_insert_then_search_then_save(csv_file_path, index_file_path);
    // printf("\n\n\n\n");
    create_and_save_then_load_then_insert_then_search(index_file_path, csv_file_path);
    // load_and_search(index_file_path, csv_file_path);
    return 0;
}
