#include "utils.h"

#include <stdlib.h>

bool write_struct_to_file(const void* st, const char* fp, const size_t size) {
    FILE *file = fopen(fp, "wb");
    if (file == NULL) {
        perror("Error opening file for writing");
        return false;
    }
    fwrite(st, size, 1, file); // Write the struct to the file
    fclose(file);
    return true;
}

bool read_struct_from_file(void* st, const char* fp, const size_t size) {
    if(!st) st = malloc(size);
    if(!st) {
        perror("Error allocating memory to load struct");
        return false;
    }
    FILE *file = fopen(fp, "rb"); // Open file for reading in binary mode
    if (file == NULL) {
        perror("Error opening file for reading");
        return false;
    }
    fread(st, size, 1, file); // Read the struct from the file
    fclose(file);
    return true;
}


void create_directory(const char *path) {
#ifdef _WIN32
    // Windows
    if (_mkdir(path) == 0) {
        printf("Directory created: %s\n", path);
    } else {
        perror("Error creating directory");
    }
#else
    // POSIX
    const mode_t mode = 0755; // Directory permissions
    if (mkdir(path, mode) == 0) {
        printf("Directory created: %s\n", path);
    } else {
        perror("Error creating directory");
    }
#endif
}

bool exists(const char* fp) {
    struct stat st = {0};
#ifdef _WIN32
    return fileAttributes != INVALID_FILE_ATTRIBUTES;
#else
    return stat(fp, &st) == 0;
#endif
}