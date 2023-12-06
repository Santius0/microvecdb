#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <omp.h>
// #include <time.h>

#include "kv_store.hpp"
#include "vector_index.hpp"

// #include "tests.h"
#include "utils.hpp"

// void fetch_and_print(const kv_store_t* kv_store, const char* key) {
//     if(strlen(key) > 0) {
//         char* value = kv_store_get(kv_store, key);
//         printf("%s => %s\n", key, value);
//         if(value) free(value);
//     } else {
//         printf("strlen(key) is 0\n");
//     }
// }

// Function to generate random float data
void generate_random_data(float* data, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
    }
}

// bool verify_kv_store(kv_store_t* store, kv_pair_t* pairs, size_t num_pairs) {
//     for (size_t i = 0; i < num_pairs; ++i) {
//         char* value = kv_store_get(store, pairs[i].key);
//         printf("found: %s => %s      .vs.     %s\n\n", pairs[i].key, value, pairs[i].value);
//         if (!value || strcmp(value, pairs[i].value) != 0) {
//             if (value) free(value); // Assuming kv_store_get allocates memory for value
//             return false;
//         }
//         free(value); // Assuming kv_store_get allocates memory for value
//     }
//     return true;
// }
//
// int test_storage() {
//     kv_store_t* store = kv_store_init("./testdb", false, false);
//
//     if (!store) {
//         fprintf(stderr, "Failed to initialize key-value store\n");
//         return 1;
//     }
//
//     // kv_pair_t pairs[] = {
//     //     {"key1", "value11"},
//     //     {"key2", "value22"},
//     //     {"key3", "value33"}
//     // };
//     kv_pair_t pairs[] = {
//         {NULL, "value11"},
//         {NULL, "value22"},
//         {NULL, "value33"}
//     };
//     size_t num_pairs = sizeof(pairs) / sizeof(pairs[0]);
//
//     if (!kv_store_put_many(store, pairs, num_pairs, true)) {
//         fprintf(stderr, "Failed to write key-value pairs to the store\n");
//         kv_store_free(store);
//         return 1;
//     }
//
//     // Verify the data was written correctly
//     if (verify_kv_store(store, pairs, num_pairs)) {
//         printf("All key-value pairs were successfully verified.\n");
//     } else {
//         fprintf(stderr, "Verification failed for key-value pairs.\n");
//         kv_store_free(store);
//         return 1;
//     }
//
//     // kv_store_free(store);
//
//     if(store) {
//
//         // char *test_file_key = "test_file_key";
//         // const bool insert_success = kv_store_put_file_name(kv_store, test_file_key, "./eye_pic.png", false);
//         // if(insert_success)
//         // printf("SUCCESS\n");
//         // else
//         // printf("FAILURE\n");
//         //
//         // char *filepath_or_buffer = "./eye_pic_fetched.png";
//         // const bool retrieve_success = kv_store_get_file_name(kv_store, test_file_key, filepath_or_buffer, true);
//         // if(retrieve_success)
//         //     printf("SUCCESS\n");
//         // else
//         //     printf("FAILURE\n");
//
//         kv_pair_t pair;
//         pair.key = "keys_for_days";
//         pair.value = "key value on 24/11/23 at 10:20AM";
//         kv_store_put(store, pair, false);
//         fetch_and_print(store, "keys_for_days");
//
//         pair.key = "keys_for_days_2";
//         pair.value = "key value on 24/11/23 at 10:20AM22222222222222";
//         kv_store_put(store, pair, false);
//         fetch_and_print(store, "keys_for_days_2");
//
//         // char *key = malloc(sizeof(char)*21);
//         // kv_store_put(store, key, "auto_inserted value", true);
//         // fetch_and_print(store, key);
//         // kv_store_put(store, key, "auto_inserted value", true);
//         // fetch_and_print(store, key);
//         // kv_store_put(store, key, "auto_inserted value", true);
//         // fetch_and_print(store, key);
//         // kv_store_put(store, key, "auto_inserted value", true);
//         // fetch_and_print(store, key);
//         // kv_store_put(store, key, "auto_inserted value", true);
//         // fetch_and_print(store, key);
//         // if(key) free(key);
//
//         kv_store_free(store);
//     }
//     return 0;
// }
//
// typedef struct test_struct {
//     int num_1;
//     float num_2;
//     vector_index_types type;
//     char* name;
// } test_struct;

#include "utils.hpp"

// #include "fasttext_wrapper.h"
#include <time.h>

#include <iostream>

#include "micrvecdb.hpp"
#include "vectorizer.hpp"
int main() {


    // mvdb::Vectorizer("./models/cc.en.300.bin", 300);
    auto* micro_vec_db = new mvdb::MicroVecDB("./test_mvdb", "test_mvdb");
    micro_vec_db->create_collection("new collection 3", 300, "./models/cc.en.300.bin");
    delete micro_vec_db;
    // auto* created = mvdb::VectorIndex::create("new_test", "./new_test", mvdb::VectorIndexType::FLAT, 300);
    // std::cout << "Created: " << std::endl << *created << std::endl;
    // created->save();
    // auto* loaded = mvdb::VectorIndex::load("new_test", "./new_test");
    // std::cout << "Loaded: " << std::endl << *loaded << std::endl;
    // delete created;
    // delete loaded;
    // clock_t start, end;
    // start = clock();
    // auto* model = fasttext_model_load("./models/cc.en.300.bin");
    // end = clock();
    // printf("create model time = %f secs\n\n", (double)(end-start)/CLOCKS_PER_SEC);


    // const char* sample_text = "The Shaka is a legendary fantasy weapon imbued with ancient magic and unparalleled craftsmanship. This mystical sword, forged in the heart of a forgotten volcano, boasts a blade that gleams with an ethereal azure light, symbolizing the boundless sky. The metal, an unknown alloy, is lighter than air yet stronger than the hardest steel, allowing it to slice through the toughest armors with ease. "
                             // "Its hilt is adorned with rare jewels and intricate engravings depicting the legends of old. The centerpiece of the hilt is a large sapphire, pulsating with a life of its own, believed to be the source of the Shaka's power. This gem grants the wielder the ability to command the winds, allowing for swift, agile movements that seem to defy gravity. Legends say that the Shaka can summon gusts strong enough to topple giants or gentle breezes to soothe the wounded. "
                             // "The Shaka's most unique feature, however, is its ability to adapt to its wielder's heart and fighting style. It can extend or shorten its blade during combat, becoming either a dagger for close quarters or a longsword for sweeping strikes. The weapon is said to resonate with the soul of its bearer, glowing brighter as it aligns with their will and intent. "
                             // "Shrouded in myths, the Shaka is not just a weapon but a symbol of ultimate power and grace, sought after by heroes and villains alike. It's said that only the pure of heart or those with a strong will can fully harness its potential without being overwhelmed by its immense power.";

    // start = clock();
    // fasttext_model_get_word_vec(sample_text, model);
    // end = clock();
    // printf("get word vector time = %f secs", (double)(end-start)/CLOCKS_PER_SEC);

    // const char* config_file_path = "/wsl.localhost/Ubuntu-22.04/home/santius/microvecdb/microvecdb/cmake-build-debug-wsl/log4c.properties";
    //
    // // Initialize log4c using the specified configuration file
    // if (log4c_init()) {
    //     fprintf(stderr, "log4c initialization failed\n");
    //     return -1;
    // }
    //
    // // Set the configuration file
    // if (log4c_rc_load(config_file_path, 0)) {
    //     fprintf(stderr, "log4c configuration failed\n");
    //     return -1;
    // }
    //
    // // Get a logger (it uses a category name, similar to log4j)
    // log4c_category_t* mycat = log4c_category_get("mycategory");
    //
    // // Log messages
    // log4c_category_log(mycat, LOG4C_PRIORITY_DEBUG, "Debug message");
    // log4c_category_log(mycat, LOG4C_PRIORITY_INFO, "Info message");
    // log4c_category_log(mycat, LOG4C_PRIORITY_ERROR, "Error message");
    //
    // // Finalize log4c
    // log4c_fini();

    // char* name = malloc(sizeof(char)*10);
    // strcpy(name, "test_name");
    // test_struct st_write = {1, 2, FLAT, name};
    // test_struct st_read;
    // write_struct_to_file(&st_write, "./test_struct.index.meta", sizeof(test_struct));
    // read_struct_from_file(&st_read, "./test_struct.index.meta", sizeof(test_struct));
    // printf("num_1: %d\nnum_2: %f\ntype: %d\nname: %s\n", st_read.num_1, st_read.num_2, st_read.type, st_read.name);

    // vector_index_t *idx = vector_index_load("test.index");
    // test_storage();
    // #pragma omp parallel
    // {
        // printf("Thread %d\n", omp_get_thread_num());
    // }

    // const char *csv_file_path = "./test_data.csv", *index_file_path = "./test.index";
    // generate_and_save_vectors(file_path);

    // create_and_insert_then_search_then_save(csv_file_path, index_file_path);
    // printf("\n\n\n\n");
    // create_and_save_then_load_then_insert_then_search(index_file_path, csv_file_path);
    // load_and_search(index_file_path, csv_file_path);


    // if(!exists("./test_collection")) create_directory("./test_collection");

    // mvdb_vector_index_t *idx = mvdb_vector_index_create("ivf_test", "./ivf_test", IVF, 100);
    // mvdb_vector_index_save(idx);
    // mvdb_vector_index_t *idx = mvdb_vector_index_load("test_collection", "./test_collection");
    // if(idx) mvdb_vector_index_free(idx);
    // idx = NULL;

    return EXIT_SUCCESS;
}
