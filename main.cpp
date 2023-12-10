#include <microvecdb.hpp>

int main() {
    auto* micro_vec_db = new mvdb::VectorDB("./test_mvdb", "test_mvdb");
    micro_vec_db->create_collection("collection1", 300, "../models/cc.en.300.bin");
    const mvdb::VectorCollection* collection = micro_vec_db->collection("collection1");

    // collection->add_data("An agile fox jumps swiftly over the sleeping dog");
    // collection->add_data("A nimble fox quickly leaps over the resting dog");
    // collection->add_data("An agile fox jumps over the sleeping canine");
    // collection->add_data("The fast brown fox jumps over the lazy hound");
    // collection->add_data("Rapidly jumping over a dog, the brown fox is swift");
    // collection->add_data("Sprinting swiftly, the red fox overcomes the resting dog");
    // collection->add_data("The quick blue fox hops over the lazy dog");
    // collection->add_data("Under a bright moon, a fox jumps over a quiet dog");
    // collection->add_data("In the forest, a brown bear climbs over a fallen log");
    // collection->add_data("Sunshine brightens the quiet forest as the deer prance away");
    // collection->add_data("hello");

    const mvdb::SearchResult sr = collection->search("The fast brown fox jumps over the lazy hound in the forest", 11, true);
    std::cout << "Search Results -\n" << sr << std::endl;
    delete micro_vec_db;
    return 0;
}
