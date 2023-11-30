#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#include <direct.h> // For Windows
#include <windows.h>
#else
#include <sys/stat.h> // For POSIX
#include <sys/types.h>
#endif

#include <stdbool.h>
#include <stdio.h>

    void mvdb_mkdir(const char *path);

    bool mvdb_exists(const char* fp);

#ifdef __cplusplus
}
#endif

#endif //UTILS_H
