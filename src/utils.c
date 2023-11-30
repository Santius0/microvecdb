#include "utils.h"

void mvdb_mkdir(const char *path) {
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

bool mvdb_exists(const char* fp) {
    struct stat st = {0};
#ifdef _WIN32
    return fileAttributes != INVALID_FILE_ATTRIBUTES;
#else
    return stat(fp, &st) == 0;
#endif
}