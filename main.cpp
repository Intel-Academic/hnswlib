
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <filesystem>

char *get_opt(char **args, int len, const std::string &option) {
    const auto end = args + len;
    char **it = std::find(args, end, option);
    if (it != end && ++it != end) {
        return *it;
    }
    return nullptr;
}

bool has_opt(char **args, int len, const std::string &option) {

    const auto end = args + len;
    char **it = std::find(args, end, option);
    if (it != end) {
        return true;
    }
    return false;
}


void sift_test1B(
        std::string &algorithm,
        int subset_size_milllions,
        int M,
        int efConstruction,
        std::vector<size_t> efs,
        int qsize,
        int n_queries,
        int k,
        std::string &ground_file,
        std::string &query_file,
        size_t repeats,
        bool permute,
        std::vector<size_t> threads,
        std::filesystem::path csv_file,
        std::string &notes
);

int main(int argc, char **argv) {
    const auto subset = get_opt(argv, argc, "--subset");
    const auto m = get_opt(argv, argc, "-m");
    const auto ef_construction = get_opt(argv, argc, "--ef-construction");
    const auto ef = get_opt(argv, argc, "--ef");

    const auto n_queries = get_opt(argv, argc, "--n_queries");
    const auto qsize = get_opt(argv, argc, "--qsize");
    const auto query_file = get_opt(argv, argc, "--query");
    const auto ground_file = get_opt(argv, argc, "--ground");
    const auto subset_millions = subset == nullptr ? 1 : std::stoi(subset);
    const auto thread_raw = get_opt(argv, argc, "--threads");
    std::vector<size_t> threads;
    if (nullptr == thread_raw) {
        threads.push_back(1);
    } else {
        std::istringstream thread_src(thread_raw);
        std::string s;
        while (getline(thread_src, s, ',')) {
            threads.push_back(std::stoi(s));
        }
    }
    const auto k = get_opt(argv, argc, "-k");
    const auto permute = has_opt(argv, argc, "--permute");
    const auto repeats = get_opt(argv, argc, "--repeat");
    const auto algorithm = get_opt(argv, argc, "--algorithm");
    if (nullptr == algorithm) {
        std::cerr << "Must specify algorithm (hnsw or hm-ann)" << std::endl;
        exit(1);
    }
    const auto output = get_opt(argv, argc, "--output");
    std::filesystem::path csv_file("results.csv");
    if (nullptr != output) {
        csv_file = output;
    }
    std::string notes;
    const auto notes_raw = get_opt(argv, argc, "--notes");
    if (nullptr != notes_raw) {
        notes = notes_raw;
    }
    const auto repeat = repeats == nullptr ? 1 : std::stoi(repeats);
    std::vector<size_t> efs;
    if (ef == nullptr) {
        std::cerr << "Must specify ef (possibly comma separated)" << std::endl;
        exit(1);
    }
    std::istringstream ef_src(ef);
    std::string s;
    while (getline(ef_src, s, ',')) {
        efs.push_back(std::stoi(s));
    }
    std::string ground = "";
    if (ground_file == nullptr) {
        ground = "bigann/gnd/idx_";
        ground += std::to_string(subset_millions);
        ground += "M.ivecs";
    } else {
        ground = ground_file;
    }
    std::string queries = "";
    if (query_file == nullptr) {
        queries = "bigann/bigann_query.bvecs";
    } else {
        queries = query_file;
    }
    std::cout << "Launching sift with params:\n"
              << "\t Algorithm: " << algorithm << "\n"
              << "\t Dataset set size: " << subset_millions << "M\n"
              << "\t M: " << m << "\n"
              << "\t ef_construction: " << ef_construction << "\n"
              << "\t ef: " << ef << "\n"
              << "\t qsize: " << qsize << "\n"
              << "\t n_queries: " << n_queries << "\n"
              << "\t k: " << k << "\n"
              << "\t query: " << queries << "\n"
              << "\t ground: " << ground << "\n"
              << "\t repeat: " << repeat << "\n"
              << "\t permute: " << permute << "\n"
              << "\t threads: ";
        std::copy(threads.begin(), threads.end(), std::ostream_iterator<size_t>(std::cout, ", "));
              std::cout << std::endl;

    std::string alg(algorithm);
    sift_test1B(
            alg,
            subset_millions,
            m == nullptr ? 1 : std::stoi(m),
            ef_construction == nullptr ? 1 : std::stoi(ef_construction),
            efs,
            qsize == nullptr ? 10000 : std::stoi(qsize),
            n_queries == nullptr ? 1 : std::stoi(n_queries),
            k == nullptr ? 1 : std::stoi(k),
            ground,
            queries,
            repeat,
            permute,
            threads,
            csv_file,
            notes
    );

    return 0;
};
