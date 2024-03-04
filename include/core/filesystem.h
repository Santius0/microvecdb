#ifndef MICROVECDB_FILESYSTEM_H
#define MICROVECDB_FILESYSTEM_H

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs {
    bool exists(const char* path) {
        #ifdef _WIN32
                DWORD dwAttrib = GetFileAttributes(path);
                return (dwAttrib != INVALID_FILE_ATTRIBUTES);
        #else
                struct stat buffer{};
                return (stat(path, &buffer) == 0);
        #endif
    }

    bool is_directory(const char* path) {
        #ifdef _WIN32
                DWORD dwAttrib = GetFileAttributes(path);
                return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
        #else
                struct stat path_stat{};
                stat(path, &path_stat);
                return S_ISDIR(path_stat.st_mode);
        #endif
    }
}


#endif //MICROVECDB_FILESYSTEM_H
