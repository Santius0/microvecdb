#ifndef MICROVECDB_FILESYSTEM_H
#define MICROVECDB_FILESYSTEM_H

#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#endif

namespace fs {

    #ifdef _WIN32
        constexpr char preferred_separator = '\\'; // Windows uses backslash as the directory separator
    #else
        constexpr char preferred_separator = '/'; // Unix-like systems use forward slash
    #endif

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
            struct stat path_stat{};
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


    inline bool create_directory(const char* path){
        #ifdef _WIN32
            return CreateDirectory(path, NULL) != 0;
        #else
            mode_t mode = 0755; // Permissions set to read+write+execute for owner, read+execute for group and others
            return mkdir(path, mode) == 0;
        #endif
    }

    inline bool create_directory(const std::string& path){
        return create_directory(path.c_str());
    }

    inline const char* get_extension(const char* path) {
        if (path == nullptr) return ""; // Check for null pointer
        const char* last_dot = strrchr(path, '.');
        if (last_dot == nullptr || last_dot == path) {
            return ""; // No extension found or dot is at the start
        }
        return last_dot;
    }

    inline const char* get_extension(const std::string& path) {
        return get_extension(path.c_str());
    }

}

#endif //MICROVECDB_FILESYSTEM_H
