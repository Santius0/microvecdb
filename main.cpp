#include <mvdb.h>

int main(){
    auto *db = new mvdb::MVDB<float>();
    delete db;
    return 0;
}