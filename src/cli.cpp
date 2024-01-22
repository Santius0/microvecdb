#include "constants.h"
#include "db.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <string>



int main(int argc, char* argv[]) {
    CLI::App app{"MicroVecDB CLI"};

    bool detailed_help = false;
    app.add_flag("-H,--HELP", detailed_help, "Print this help menu(detailed) and exit");

    CLI::App *open_cmd = app.add_subcommand("open", "Open a database");
    std::string dbname, dbpath;
    uint64_t dims;
    mvdb::IndexType index_type;
    mvdb::VectorizerModelType vec_model;
    open_cmd->add_option("-d,--dbname", dbname, "Database name")->required();
    open_cmd->add_option("-p,--dbpath", dbpath, "Database path")->required();
    open_cmd->add_option("--dims", dims, "Dimensions")->required();
    open_cmd->add_option("--index_type", index_type, "Index Type")->required();
    open_cmd->add_option("--vec_model", vec_model, "Vectorizer Model Type")->required();

    // Subcommand for adding vectors
    CLI::App *add_cmd = app.add_subcommand("add", "Add a vector");
    std::vector<float> vector;
    add_cmd->add_option("-v,--vector", vector, "Vector to add")->required();

    // Subcommand for searching with a vector
    CLI::App *search_cmd = app.add_subcommand("search", "Search with a vector");
    std::vector<float> query_vector;
    long k;
    bool ret_data;
    search_cmd->add_option("-q,--query", query_vector, "Query vector")->required();
    search_cmd->add_option("-k,--top_k", k, "Number of top results")->required();
    search_cmd->add_option("-r,--return_data", ret_data, "Return data flag")->required();

    CLI11_PARSE(app, argc, argv);

    if(detailed_help) app.exit(CLI::CallForAllHelp());

    if(*open_cmd) {
        mvdb::DB db(dbname, dbpath, dims, index_type, vec_model);
    }

    if (*add_cmd) {
        mvdb::DB db(dbname, dbpath, dims, index_type, vec_model);
        // Handle add vector command
        uint64_t* keys = db.add_vector(vector.size(), vector.data());
        if (keys) {
            std::cout << "Vector added. Keys: ";
            for (size_t i = 0; i < vector.size(); ++i) {
                std::cout << keys[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to add vector." << std::endl;
        }
    } else if (*search_cmd) {
        // Handle search with vector command
        mvdb::DB db(dbname, dbpath, dims, index_type, vec_model);
        mvdb::SearchResult* result = db.search_with_vector(query_vector.size(), query_vector.data(), k, ret_data);
        // Process and display search results
        // ...
    } else {
        app.exit(CLI::CallForHelp());
    }

    return 0;
}
