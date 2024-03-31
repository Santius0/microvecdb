#ifndef MICROVECDB_FILESYSTEM_H
#define MICROVECDB_FILESYSTEM_H

#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#endif

namespace fs {

    inline bool exists(const char* path) {
        #ifdef _WIN32
                DWORD dwAttrib = GetFileAttributes(path);
                return (dwAttrib != INVALID_FILE_ATTRIBUTES);
        #else
                struct stat buffer{};
                return (stat(path, &buffer) == 0);
        #endif
    }

    inline bool exists(const std::string& path) {
        return exists(path.c_str());
    }

    inline bool is_directory(const char* path) {
        #ifdef _WIN32
                DWORD dwAttrib = GetFileAttributes(path);
                return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
        #else
                struct stat path_stat{};
                stat(path, &path_stat);
                return S_ISDIR(path_stat.st_mode);
        #endif
    }

    inline bool is_directory(const std::string& path) {
        return is_directory(path.c_str());
    }

    inline bool remove(const char* path) {
        #ifdef _WIN32
            DWORD dwAttrib = GetFileAttributes(path);

            if(dwAttrib == INVALID_FILE_ATTRIBUTES)
                return false; // Path does not exist

            if(dwAttrib & FILE_ATTRIBUTE_DIRECTORY)
                return RemoveDirectory(path) != 0; // Remove directory
            else
                return DeleteFile(path) != 0; // Remove file
        #else
            struct stat path_stat;
            stat(path, &path_stat);

            if(S_ISDIR(path_stat.st_mode))
                return rmdir(path) == 0; // Remove directory
            else
                return unlink(path) == 0; // Remove file
        #endif
    }

    inline bool remove(const std::string& path) {
        return remove(path.c_str());
    }

}

#endif //MICROVECDB_FILESYSTEM_H
