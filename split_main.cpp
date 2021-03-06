#include <deque>
#include "hnswlib/hm_ann.h"

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



std::string sift_index_name(size_t subset_millions, size_t ef_construction, size_t M) {
    std::ostringstream prefix;
    prefix << "sift1b_" << subset_millions << "m";
    auto p = prefix.str();
    return index_name(p, ef_construction, M);
}

int main(int argc, char **argv) {
    using namespace hnswlib;
    const auto alg = get_opt(argv, argc, "--algorithm");
    if (nullptr == alg) {
        std::cerr << "Must specify algorithm (hnsw or hm-ann)" << std::endl;
        exit(1);
    }
    std::string algorithm(alg);
    const auto input = get_opt(argv, argc, "--input");
    if (nullptr == input) {
        std::cerr << "--input required" << std::endl;
        exit(1);
    }
    std::string path_index(input);
    const auto prefix = get_opt(argv, argc, "--prefix");
    std::string file_prefix = "";
    if (prefix == nullptr) {
        file_prefix = "split-";
    } else {
        file_prefix = prefix;
    }
    std::cout << "Splitting index at " << path_index <<  "\n"
              << "\t with output prefix: " << file_prefix << "\n"
              << std::endl;

    size_t vecdim = 128;
    L2SpaceI l2space(vecdim);

    HierarchicalNSW<int> *appr_alg;
    auto path_l0 = level0_name(path_index);
    if (std::filesystem::exists(path_index)) {
        if (algorithm == "hnsw") {
            appr_alg = new HierarchicalNSW<int>(&l2space, path_index, false, 0, path_l0);
        } else if (algorithm == "hm-ann"){
            appr_alg = new HmAnn<int>(&l2space, path_index, false, 0, path_l0);
        }
        std::cout << "Writing split index";
        appr_alg->writeIndex(file_prefix + path_index, false, true);
    } else {
        std::cout << "No index  " << path_index << "found\n";
    }
    return 0;
};
