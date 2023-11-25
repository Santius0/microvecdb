#include "vector_index.h"

#include <stdlib.h>
#include <stdio.h>

void vector_index_test() {
    // Define the dimensionality of the vectors and the number of vectors
    int d = 64;  // dimension
    int nb = 10000;  // database size
    int nq = 100;  // number of queries
    int k = 4;  // number of nearest neighbors

    // Allocate memory for the vectors
    float *xb = (float*)malloc(d * nb * sizeof(float));
    float *xq = (float*)malloc(d * nq * sizeof(float));

    // Initialize the vectors (Here, just random values for example)
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
    }

    // Create a FlatL2 index
    FaissIndex* index;
    faiss_IndexFlatL2_new(&index);

    // Add vectors to the index
    faiss_Index_add(index, nb, xb);

    // Allocate memory for search results
    int64_t *I = (int64_t*)malloc(k * nq * sizeof(int64_t));
    float *D = (float*)malloc(k * nq * sizeof(float));

    // Search the index
    faiss_Index_search(index, nq, xq, k, D, I);

    // Display the results
    printf("Search results:\n");
    for(int i = 0; i < nq; i++) {
        printf("Query %d:\n", i);
        for(int j = 0; j < k; j++)
            printf("    %3ld  %.6f\n", I[i * k + j], D[i * k + j]);
    }

    // Clean up
    free(xb);
    free(xq);
    free(I);
    free(D);
    faiss_Index_free(index);
}
