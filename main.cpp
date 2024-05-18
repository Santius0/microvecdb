#include <mvdb.h>

int main(){
    auto *db = new mvdb::MVDB<float>();
    delete db;
    int a = 1, b = 2;
#ifdef FAISS
    if(a == 1){
        std::cout << "1";
    }
    else
#endif
    if(a == 1){
            std::cout << "dd";
    }
    return 0;
}