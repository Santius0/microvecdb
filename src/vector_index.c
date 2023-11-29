#include "vector_index.h"
#include "utils.h"
#include "constants.h"

#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/Clustering_c.h>
#include <faiss/c_api/index_io_c.h>
#include <stdlib.h>
#include <string.h>


vector_index_t* create_vector_index(const char* name, const char* dir, const vector_index_types type, const uint64_t dims) {
    vector_index_t* vi = malloc(sizeof(vector_index_t));
    if (!vi) return NULL;

    vi->name = strdup(name);
    vi->dir = strdup(dir);
    vi->dims = dims;
    vi->type = type;

    if(!name || !dir || dims == 0) return NULL;

    switch (type) {
        case IVF:
            // For simplicity, using a flat quantizer and a fixed nlist value
            const int nlist = 100;
            FaissIndex* quantizer;
            faiss_IndexFlatL2_new_with(&quantizer, dims);
            faiss_IndexIVFFlat_new_with_metric(&vi->faiss_index, quantizer, dims, nlist, METRIC_L2);
            faiss_Index_free(quantizer); // The index copies the quantizer
        default:
            faiss_IndexFlatL2_new_with(&vi->faiss_index, dims);
    }
    return vi;
}

void free_vector_index(vector_index_t* vi) {
    if (!vi) return;
    if (vi->faiss_index) faiss_Index_free(vi->faiss_index);
    free(vi);
}

bool vector_index_add(const vector_index_t* vi, const size_t n, const float* data){
    if (!vi || !vi->faiss_index || !data) return false;
    faiss_Index_add(vi->faiss_index, n, data);
    return true;
}

bool vector_index_remove(const vector_index_t* vi, const size_t n, const FaissIDSelector* ids) {
    if (!vi || !vi->faiss_index || !ids) return false;
    faiss_Index_remove_ids(vi->faiss_index, ids, n);
    return true;
}

bool vector_index_save(const vector_index_t* vi) {
    if (!vi || !vi->faiss_index) return false;

    const size_t path_preamble_len = strlen(vi->dir) + strlen(vi->name);
    // paths are dir/collection_name.index and dir/collection_name.index.meta
    // the extra 1 allocated is for the "/"
    char *index_path = malloc(sizeof(char) * (path_preamble_len + INDEX_EXT_LEN + 1));
    if(!index_path) return false;
    char *index_meta_path = malloc(sizeof(char) * (path_preamble_len + INDEX_META_EXT_LEN + 1));
    if(!index_meta_path) return false;

    sprintf(index_path, "%s/%s%s",vi->dir, vi->name, INDEX_EXT);
    sprintf(index_meta_path, "%s/%s%s",vi->dir, vi->name, INDEX_META_EXT);

    printf("Index saving to: %s\n\n", index_path);
    printf("Index Meta saving to: %s\n\n", index_meta_path);

    // TODO: verify save happened successfully
    faiss_write_index_fname(vi->faiss_index, index_path);
    write_struct_to_file((void*)vi, index_meta_path, sizeof(vi));

    if(index_path) free(index_path);
    if(index_meta_path) free(index_meta_path);

    return true;
}

vector_index_t* vector_index_load(const char* path) {
    if (!path) return NULL;
    vector_index_t* vi = malloc(sizeof(vector_index_t));
    if (!vi) return NULL;

    faiss_read_index_fname(path, 0, &vi->faiss_index);
    if (!vi->faiss_index) {
        free_vector_index(vi);
        return NULL;
    }
    vi->type = FLAT;
    vi->dims = 100;
    return vi;
}
