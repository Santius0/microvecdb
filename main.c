#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kv_store.h"

void fetch_and_print(const kv_store_t* kv_store, const char* key) {
    char* value = kv_store_get(kv_store, key);
    printf("%s => %s", key, value);
    if(value) free(value);
}

int main() {
    kv_store_t* kv_store = kv_store_init("./testdb");

    // kv_store_put(kv_store, "key1", "key value on 24/11/23 at 10:20AM");
    // kv_store_put(kv_store, "key10", "key value on 24/11/23 at 10:20AM");

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


    // ConfluxStorage *data_storage = ConfluxStorage::getInstance("./testdb");
    // ConfluxIndex *vector_index = ConfluxIndex::getInstance(40, "test.ann");
    //
    // // data_storage->put(data_storage->get_latest_sequence_number(), "value stored on 19/11/2023 at 4:16 pm");
    // std::string fetched;
    // data_storage->get("key1", fetched);
    // std::cout << fetched << std::endl;
    //
    // // Dimensionality of the vectors
    // int d = 64;
    //
    // // Create a flat (brute force) index
    // faiss::IndexFlatL2 index(d);
    //
    // // Number of vectors to add
    // int n = 100;
    //
    // // Make some random vectors to add to the index
    // std::vector<float> xb(d * n);
    // for(int i = 0; i < n; i++) {
    //     for(int j = 0; j < d; j++)
    //         xb[d * i + j] = drand48();
    // }
    //
    // // Add vectors to the index
    // index.add(n, xb.data());
    //
    // // Number of nearest neighbors to search for
    // int k = 4;
    //
    // // Make a random query vector
    // std::vector<float> xq(d);
    // for(int j = 0; j < d; j++)
    //     xq[j] = drand48();
    //
    // // Vectors to hold search results
    // std::vector<float> distances(k);
    // std::vector<int64_t> labels(k);
    //
    // // Perform the search
    // index.search(1, xq.data(), k, distances.data(), labels.data());
    //
    // // Display the results
    // std::cout << "Search results:" << std::endl;
    // for(int i = 0; i < k; i++)
    //     std::cout << "Index: " << labels[i] << ", Distance: " << distances[i] << std::endl;
    //
    // ConfluxStorage::destroyInstance();
    // ConfluxIndex::destroyInstance();
    // return 0;
}
