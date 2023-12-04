// #include "vector_collection.h"
//
// #include <string.h>
// #include <stdlib.h>
//
//
// vector_collection_t* vector_collection_init(const char* name, const char* path){
//     vector_collection_t *collection = malloc(sizeof(vector_collection_t));
//     if(!collection) return NULL;
//
//     collection->name = strdup(name);
//     if(!collection->name) {
//         vector_collection_free(collection);
//         return NULL;
//     }
//     collection->path = strdup(path);
//     if(!collection->path) {
//         vector_collection_free(collection);
//         return NULL;
//     }
//
//     return collection;
// }
//
// vector_collection_t* vector_collection_create(const char* name, const char* path, const bool overwrite) {
//     vector_collection_t* collection = vector_collection_init(name, path);
//
//     // initialise kv_store
//     collection->kv_store = kv_store_init(path, true, overwrite);
//     if(!collection->kv_store) vector_collection_free(collection);
//
//     // initialise vector_store
//     // place vector_store init code here
//
//     // if(!collection->vector_store) vector_collection_free(collection);
//
//     return collection;
// }
//
// vector_collection_t* vector_collection_load(const char* name, const char* path) {
//     vector_collection_t* collection = vector_collection_init(name, path);
//
//     // initialise kv_store
//     collection->kv_store = kv_store_init(path, false, false);
//     if(!collection->kv_store) vector_collection_free(collection);
//
//     // initialise vector_store
//     // place vector_store init code here
//
//     // if(!collection->vector_store) vector_collection_free(collection);
//
//     return collection;
// }
//
// void vector_collection_free(vector_collection_t* collection) {
//     if(collection) {
//         if(collection->name) free(collection->name);
//         if(collection->path) free(collection->path);
//         if(collection->kv_store) kv_store_free(collection->kv_store);
//         free(collection);
//         collection = NULL;
//     }
// }
//
// bool vector_collection_add(vector_collection_t* collection, const char* key, const float* vector, size_t vector_length) {
//     return true;
// }
//
