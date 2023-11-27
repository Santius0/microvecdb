#include "vector_index.h"
#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/Clustering_c.h>
#include <faiss/c_api/index_io_c.h>
#include <stdlib.h>
#include <string.h>

vector_index* create_vector_index(vector_index_types type, uint64_t dims) {
    vector_index* vi = malloc(sizeof(vector_index));
    if (!vi) return NULL;

    vi->dims = dims;
    vi->type = type;

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

void free_vector_index(vector_index* vi) {
    if (!vi) return;
    if (vi->faiss_index) faiss_Index_free(vi->faiss_index);
    free(vi);
}

bool vector_index_add(vector_index* vi, size_t n, const float* data) {
    if (!vi || !vi->faiss_index || !data) return false;
    faiss_Index_add(vi->faiss_index, n, data);
    return true;
}

bool vector_index_remove(vector_index* vi, size_t n, const FaissIDSelector* ids) {
    if (!vi || !vi->faiss_index || !ids) return false;
    faiss_Index_remove_ids(vi->faiss_index, ids, n);
    return true;
}

bool vector_index_save(const vector_index* vi, const char* path) {
    if (!vi || !vi->faiss_index || !path) return false;
    faiss_write_index_fname(vi->faiss_index, path);
    return true;
}

vector_index* vector_index_load(const char* path) {
    if (!path) return NULL;
    vector_index* vi = malloc(sizeof(vector_index));
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
