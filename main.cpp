#include <string>
#include <algorithm>
#include <iostream>

char* get_opt(char** args, int len, const std::string &option) {
	const auto end = args + len;
	char ** it = std::find(args, end, option);
	if (it != end && ++it != end ) {
		return *it;
	}
	return nullptr;
}

bool has_opt(char**args, int len, const std::string &option) {

	const auto end = args + len;
	char ** it = std::find(args, end, option);
	if (it != end) {
		return true;
	}
	return false;
}


void sift_test1B(
		    int subset_size_milllions,
		    int M, 
		    int efConstruction,
		    int ef,
		    int n_queries,
		    int k,
		    std::string &ground_file,
		    std::string &query_file
		);
int main(int argc, char** argv) {
	const auto subset = get_opt(argv, argc, "--subset");
	const auto m = get_opt(argv, argc, "-m");
	const auto ef_construction = get_opt(argv, argc, "--ef-construction");
	const auto ef = get_opt(argv, argc, "--ef");
	const auto n_queries = get_opt(argv, argc, "--n_queries");
	const auto query_file = get_opt(argv, argc, "--query");
	const auto ground_file = get_opt(argv, argc, "--ground");
	const auto subset_millions = subset == nullptr ? 1 : std::stoi(subset);
	const auto k = get_opt(argv, argc, "-k");
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
		<< "\t Dataset set size: " << subset_millions << "M\n"
		<< "\t M: " << m << "\n"
		<< "\t ef_construction: " << ef_construction << "\n"
		<< "\t ef: " << ef << "\n"
		<< "\t n_queries: " << n_queries << "\n"
		<< "\t k: " << k << "\n"
		<< "\t query: " << queries << "\n"
		<< "\t ground: " << ground << "\n"
		<< std::endl;



    sift_test1B(
		    subset_millions,
		    m == nullptr ? 1 : std::stoi(m),
		    ef_construction == nullptr ? 1 : std::stoi(ef_construction),
		    ef == nullptr ? 1 : std::stoi(ef),
		    n_queries == nullptr ? 1 : std::stoi(n_queries),
		    k == nullptr ? 1 : std::stoi(k),
		    ground,
		    queries
		    );

    return 0;
};
