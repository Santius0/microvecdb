#ifndef TESTS_H
#define TESTS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vector_index.h"

#define NUM_VECTORS 100
#define DIMENSIONS 94
#define TOP_K 100
#define MAX_LINE_LENGTH (DIMENSIONS * 10)

static void generate_and_save_vectors(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            fprintf(file, "%.2f", (float)rand() / RAND_MAX); // Generate a random float
            if (j < DIMENSIONS - 1) {
                fprintf(file, ","); // Comma for CSV format
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Function to parse a line into a vector
static bool parse_vector(const char* line, float* vector, size_t dims) {
    const char* token = strtok((char*)line, ",");
    for (size_t i = 0; i < dims; ++i) {
        if (token == NULL) {
            return false;
        }
        vector[i] = atof(token);
        token = strtok(NULL, ",");
    }
    return true;
}

static void create_and_insert_then_search_then_save(const char* filename, const char* index_filename) {
    srand(42);
    // Step 1: Create a new vector index
    mvdb_vector_index_t* vi = mvdb_vector_index_create("./test_collection", "./test_collection", FLAT, DIMENSIONS);
    if (!vi) {
        printf("Failed to create vector index.\n");
        return;
    }

    // Step 2: Insert data into the index
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        mvdb_vector_index_free(vi);
        return;
    }

    char line[MAX_LINE_LENGTH];
    float vector[DIMENSIONS];
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (parse_vector(line, vector, DIMENSIONS)) {
            if (!mvdb_vector_index_add(vi, 1, vector)) {
                printf("Failed to add vector to index.\n");
            }
        }
    }
    fclose(file);

    // Step 3: Search for a random vector
    file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        mvdb_vector_index_free(vi);
        return;
    }

    // Determine number of vectors in file
    size_t num_vectors = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        num_vectors++;
    }
    fseek(file, 0, SEEK_SET); // Reset file pointer to the beginning

    // Select a random vector
    // size_t random_line = rand() % num_vectors;
    size_t random_line = 66;
    for (size_t i = 0; i <= random_line; ++i) {
        if (!fgets(line, MAX_LINE_LENGTH, file)) {
            printf("Error reading file.\n");
            fclose(file);
            mvdb_vector_index_free(vi);
            return;
        }
    }

    if (!parse_vector(line, vector, DIMENSIONS)) {
        printf("Error parsing vector.\n");
        fclose(file);
        mvdb_vector_index_free(vi);
        return;
    }
    fclose(file);

    // Perform the search
    int64_t* labels = malloc(sizeof(int64_t) * TOP_K);
    float* distances = malloc(sizeof(float) * TOP_K);
    faiss_Index_search(vi->faiss_index, 1, vector, TOP_K, distances, labels);

    // Print results
    printf("Searched for vector at line %zu\n", random_line + 1);
    for (int i = 0; i < TOP_K; ++i) {
        printf("Index: %lld, Distance: %f\n", labels[i], distances[i]);
    }

    if (!mvdb_vector_index_save(vi)) {
        printf("Failed to save vector index.\n");
        fclose(file);
        mvdb_vector_index_free(vi);
        return;
    }

    // Cleanup
    free(labels);
    free(distances);
    mvdb_vector_index_free(vi);
}

static void create_and_save_then_load_then_insert_then_search(const char* index_filename, const char* csv_filename) {
    srand(42);
    // Step 1: Create a new vector index
    mvdb_vector_index_t* vi = mvdb_vector_index_create("test_collection", "./test_collection", IVF, DIMENSIONS);
    if (!vi) {
        printf("Failed to create vector index.\n");
        return;
    }

    // Step 2: Insert half of the CSV data
    FILE* file = fopen(csv_filename, "r");
    if (!file) {
        perror("Error opening CSV file");
        mvdb_vector_index_free(vi);
        return;
    }

    size_t total_vectors = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        total_vectors++;
    }
    fseek(file, 0, SEEK_SET);

    size_t half_vectors = total_vectors / 2;
    float vector[DIMENSIONS];
    for (size_t i = 0; i < half_vectors; ++i) {
        if (fgets(line, MAX_LINE_LENGTH, file) && parse_vector(line, vector, DIMENSIONS)) {
            if (!mvdb_vector_index_add(vi, 1, vector)) {
                printf("Failed to add vector to index.\n");
            }
        }
    }

    // Step 3: Save the vector index
    if (!mvdb_vector_index_save(vi)) {
        printf("Failed to save vector index.\n");
        fclose(file);
        mvdb_vector_index_free(vi);
        return;
    }

    // Step 4: Delete the original vector index instance
    mvdb_vector_index_free(vi);

    // Step 5: Load the saved vector index
    vi = mvdb_vector_index_load("test_collection", "./test_collection");
    if (!vi) {
        printf("Failed to load vector index.\n");
        fclose(file);
        return;
    }

    // Step 6: Insert the remaining half of the CSV data
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (parse_vector(line, vector, DIMENSIONS)) {
            if (!mvdb_vector_index_add(vi, 1, vector)) {
                printf("Failed to add vector to index.\n");
            }
        }
    }
    fclose(file);

    // Step 7: Search for a random vector from CSV file
    file = fopen(csv_filename, "r");
    if (!file) {
        perror("Error opening CSV file");
        mvdb_vector_index_free(vi);
        return;
    }

    size_t random_line = rand() % total_vectors;
    fseek(file, 0, SEEK_SET);
    for (size_t i = 0; i <= random_line; ++i) {
        if (!fgets(line, MAX_LINE_LENGTH, file)) {
            printf("Error reading file.\n");
            fclose(file);
            mvdb_vector_index_free(vi);
            return;
        }
    }

    if (!parse_vector(line, vector, DIMENSIONS)) {
        printf("Error parsing vector.\n");
        fclose(file);
        mvdb_vector_index_free(vi);
        return;
    }
    fclose(file);

    // Perform the search
    int64_t* labels = malloc(sizeof(int64_t) * TOP_K);
    float* distances = malloc(sizeof(float) * TOP_K);
    faiss_Index_search(vi->faiss_index, 1, vector, TOP_K, distances, labels);

    // Print results
    printf("Searched for vector at line %zu\n", random_line + 1);
    for (int i = 0; i < TOP_K; ++i) {
        printf("Index: %lld, Distance: %f\n", labels[i], distances[i]);
    }

    // Cleanup
    free(labels);
    free(distances);
    mvdb_vector_index_free(vi);
}

static void load_and_search(const char* index_filename, const char* csv_filename) {
    srand(42);
    // Step 1: Load the saved vector index
    mvdb_vector_index_t* vi = mvdb_vector_index_load("test_collection", "./test_collection");
    if (!vi) {
        printf("Failed to load vector index.\n");
        return;
    }

    // Step 2: Read and select a random vector from the CSV file
    FILE* file = fopen(csv_filename, "r");
    if (!file) {
        perror("Error opening CSV file");
        mvdb_vector_index_free(vi);
        return;
    }

    size_t num_vectors = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        num_vectors++;
    }
    if (num_vectors == 0) {
        printf("No vectors found in the file.\n");
        fclose(file);
        mvdb_vector_index_free(vi);
        return;
    }

    // size_t random_line = rand() % num_vectors;
    size_t random_line = 66;
    fseek(file, 0, SEEK_SET); // Reset file pointer to the beginning

    for (size_t i = 0; i <= random_line; ++i) {
        if (!fgets(line, MAX_LINE_LENGTH, file)) {
            printf("Error reading file.\n");
            fclose(file);
            mvdb_vector_index_free(vi);
            return;
        }
    }

    // add new vector to loaded index
    // FILE* file_to_add = fopen(csv_filename, "r");
    // if (!file_to_add) {
    //     perror("Error opening file");
    //     mvdb_vector_index_free(vi);
    //     return;
    // }
    //
    // char line_to_add[MAX_LINE_LENGTH];
    // float vector_to_add[DIMENSIONS];
    // while (fgets(line_to_add, MAX_LINE_LENGTH, file_to_add)) {
    //     if (parse_vector(line_to_add, vector_to_add, DIMENSIONS)) {
    //         if (!mvdb_vector_index_add(vi, 1, vector_to_add)) {
    //             printf("Failed to add vector to index.\n");
    //         }
    //     }
    // }
    // fclose(file_to_add);
    // vector_index_save(vi, index_filename);
    // ----------------------------------------------------------------------- //

    float query_vector[DIMENSIONS];
    if (!parse_vector(line, query_vector, DIMENSIONS)) {
        printf("Error parsing vector.\n");
        fclose(file);
        mvdb_vector_index_free(vi);
        return;
    }
    fclose(file);

    // Step 3: Search for the random vector in the loaded index
    int64_t* labels = malloc(sizeof(int64_t) * TOP_K);
    float* distances = malloc(sizeof(float) * TOP_K);
    faiss_Index_search(vi->faiss_index, 1, query_vector, TOP_K, distances, labels);

    // Print results
    printf("Searched for vector at line %zu\n", random_line + 1);
    for (int i = 0; i < TOP_K; ++i) {
        printf("Index: %lld, Distance: %f\n", labels[i], distances[i]);
    }

    // Cleanup
    free(labels);
    free(distances);
    mvdb_vector_index_free(vi);
}

#endif //TESTS_H
