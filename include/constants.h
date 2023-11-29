#ifndef CONTSTANTS_H
#define CONTSTANTS_H

// Index store contants
#define INDEX_EXT ".index"
#define INDEX_EXT_LEN strlen(INDEX_EXT)
#define INDEX_META_EXT ".index.meta"
#define INDEX_META_EXT_LEN strlen(INDEX_META_EXT)

// Data store contsants
#define MAX_KEY_CHARS 21
#define MAX_KEY_SIZE_BYTES sizeof(char) * MAX_KEY_CHARS // (2^64 - 1) is a number with 20 digits

#endif //CONTSTANTS_H
