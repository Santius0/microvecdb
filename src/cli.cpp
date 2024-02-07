#include "constants.h"
#include "db.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <string>


int main(int argc, char* argv[]) {
    CLI::App app{"MicroVecDB CLI"};

    // Detailed help command
    bool detailed_help = false;
    app.add_flag("-H,--HELP", detailed_help, "Print this help menu(detailed) and exit");

    bool version = false;
    app.add_flag("-v,--version", version, "Print application version");

    // Subcommand for opening DB
    std::string dbname, dbpath = ".";
    uint64_t dims;
    mvdb::IndexType index_type = mvdb::IndexType::FAISS_FLAT;
    mvdb::VectorizerModelType vec_model = mvdb::VectorizerModelType::FASTTEXT;
    std::vector<float> vector;
    std::vector<float> query_vector;
    long k = 5;
    bool ret_data = false;
    bool detach = false;

    CLI::App *open_cmd = app.add_subcommand("open", "Open a database");
    open_cmd->add_option("-d,--dbname", dbname, "Database name")->required();
    open_cmd->add_option("-p,--dbpath", dbpath, "Database path");
    open_cmd->add_option("--dims", dims, "Dimensions")->required();
    open_cmd->add_option("--index_type", index_type, "Index Type");
    open_cmd->add_option("--vec_model", vec_model, "Vectorizer Model Type");

    // Subcommand for adding vectors
    CLI::App *add_cmd = app.add_subcommand("add", "Add a vector");
    add_cmd->add_option("-p,--dbpath", dbpath, "Database path")->required();
    add_cmd->add_option("-v,--vector", vector, "Vector to add")->required();

    // Subcommand for searching with a vector
    CLI::App *search_cmd = app.add_subcommand("search", "Search with a vector");
    search_cmd->add_option("-p,--dbpath", dbpath, "Database path")->required();
    search_cmd->add_option("-q,--query", query_vector, "Query vector")->required();
    search_cmd->add_option("-k,--top_k", k, "Number of top results");
    search_cmd->add_option("-r,--return_data", ret_data, "Return data flag");

    CLI::App *start_cmd = app.add_subcommand("start", "Start database server");
    start_cmd->add_flag("-d,--detach", detach, "Detach for silent run in background");

    CLI11_PARSE(app, argc, argv);

    if(detailed_help)
        app.exit(CLI::CallForAllHelp());
    else if(version)
        std::cout << VERSION << std::endl;
    else if(*open_cmd) {
        mvdb::DB db(dbname, dbpath, dims, index_type, vec_model);
        db.save();
    } else if (*add_cmd) {
        mvdb::DB db(dbname, dbpath, dims, index_type, vec_model);
        mvdb::idx_t* keys = db.add_vector(vector.size(), vector.data());
        if (keys) {
            std::cout << "Vector added. Keys: ";
            for (size_t i = 0; i < vector.size(); ++i)
                std::cout << keys[i] << " ";
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to add vector." << std::endl;
        }
    } else if (*search_cmd) {
        mvdb::DB db(dbname, dbpath, dims, index_type, vec_model);
        auto *ids = new mvdb::idx_t[k];
        auto *distances = new mvdb::value_t[k];
        db.search_with_vector(query_vector.size(), query_vector.data(), k, ids, distances);
        for (size_t i = 0; i < vector.size()*k; ++i)
            std::cout << ids[i] << " => " << distances[i] << std::endl;
        delete[] ids;
        delete[] distances;
    } else if(*start_cmd){
        if(detach) std::cout << "start server detached";
        else std::cout << "start server not detached";
    }
    else {
        app.exit(CLI::CallForHelp());
    }
    return 0;
}
