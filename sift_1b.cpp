#include <iostream>
#include <fstream>
#include <queue>

#define HNSW_MMAP

#include "hnswlib/hm_ann.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;




/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


static void
get_gt(unsigned int *massQA, unsigned char *massQ, unsigned char *mass, size_t vecsize, size_t qsize, L2SpaceI &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<int, labeltype >>> &answers, size_t k) {


    (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}


struct TestResult {
	size_t k;
	float recall;
	vector<Times> times;
};

static TestResult
test_approx(unsigned char *massQ, size_t qsize, size_t vecsize, size_t n_queries, HierarchicalNSW<int> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<int, labeltype >>> &answers, size_t k, bool permute) {
	TestResult results = {};
	results.k = k;
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    vector<size_t> permutation(qsize);
    //cout << "permuting" << endl;
    std::iota(permutation.begin(), permutation.end(), 0);
    std::random_device rng;
    std::mt19937 urng(rng());
    if (permute) {
	    std::shuffle(permutation.begin(), permutation.end(), urng);
    }
    //cout << "running queries" << endl;
    for (int i = 0; i < n_queries; i++) {
	size_t offset = permutation[i];
        StopW query_time;
        QueryResult result = appr_alg.searchKnn(massQ + vecdim * offset, k);
	result.times.total_micros = query_time.getElapsedTimeMicro();
	results.times.push_back(result.times);
        std::priority_queue<std::pair<int, labeltype >> gt(answers[permutation[i]]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {


            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.result.size()) {
            if (g.find(result.result.top().second) != g.end()) {

                correct++;
            } else {
            }
            result.result.pop();
        }

    }
    results.recall = 1.0f * correct / total;
    return results;
}

struct Stats {
	long double mean;
	long double total;
	long double std;
	long double three_nines;
	long double min;
	long double max;
};

Stats calculate_stats(std::vector<float> &floats) {
	// Note - floats will be sorted in place at the end
	Stats stats = {};
	//cout << "calculating total" << endl;
	long double squares = 0;
	stats.min = std::numeric_limits<long double>::max();
	stats.max = std::numeric_limits<long double>::min();
	for (auto f: floats) {
		stats.total += f;
		if (f < stats.min) {
			stats.min = f;
		}
		if (f > stats.max) {
			stats.max = f;
		}
	}
	//cout << "calculating mean" << endl;
	stats.mean = stats.total / (long double)floats.size();

	//cout << "calculating squares" << endl;
        for (auto f: floats) {
            squares += std::pow(f - stats.mean, 2);
        }

	//cout << "calculating sqrt" << endl;
        stats.std = std::sqrt(squares / (long double)floats.size());
	//cout << "sorting" << endl;
        std::sort(floats.begin(), floats.end());
	//cout << "three nines" << endl;
	if (floats.size() >= 1000) {
		auto three_nines = std::ceil((99.9/100.0) * floats.size());
		stats.three_nines = floats[three_nines];
	} else {
		stats.three_nines = std::numeric_limits<long double>::quiet_NaN();
	}

	return stats;
}

static void
test_vs_recall(unsigned char *massQ, size_t qsize, size_t vecsize, size_t n_queries, HierarchicalNSW<int> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<int, labeltype >>> &answers, size_t k, vector<size_t> efs, 
	       size_t repeats, bool permute) {
    //for (int i = k; i < 30; i++) {
        //efs.push_back(i);
    //}
    //for (int i = 30; i < 100; i += 10) {
        //efs.push_back(i);
    //}
    //for (int i = 100; i < 500; i += 40) {
        //efs.push_back(i);
    //}
	cout << "| run | #q| ef | " << k << "-recall | qps " 
		<< "| query_us | hier_us | L0_us "
		<< "| q_std | h_std | L0_std "
		<< "| q_999 | h_999 | L0_999 "
		<< "| q_min | h_min | L0_min "
		<< "| q_max | h_max | L0_max "
		<< "| batch_us | batch_hier_us | batch_L0_us" << endl;
	cout << "|--|--|--|--|--||||||---|-------|-----|----------|----------|---------|---------|---------|---------|---------|-------|-------|--------|"<< endl;
	for (size_t run = 1; run <= repeats; run++) {
		//cout << "starting run " << run << endl;
	    for (size_t ef : efs) {
		//cout << "starting ef " << ef << endl;
		appr_alg.setEf(ef);
		// StopW stopw = StopW();
		//cout << "Running queries" << endl;
		auto test_result = test_approx(massQ, qsize, vecsize, n_queries, appr_alg, vecdim, answers, k, permute);
		//cout << "Collecting stats" << endl;
		vector<float> l0_times;
		std::transform(test_result.times.begin(), test_result.times.end(), std::back_inserter(l0_times),
				[](Times times) { return times.l0_micros; });
		auto l0_stats = calculate_stats(l0_times);
		vector<float> ln_times;
		std::transform(test_result.times.begin(), test_result.times.end(), std::back_inserter(ln_times),
				[](Times times) { return times.ln_micros; });
		auto ln_stats = calculate_stats(ln_times);
		vector<float> total_times;
		std::transform(test_result.times.begin(), test_result.times.end(), std::back_inserter(total_times),
				[](Times times) { return times.total_micros; });
		auto total_stats = calculate_stats(total_times);

		auto qps = n_queries / (total_stats.total / (1000.0 * 1000.0));
		cout << " | " << run << "|" << n_queries << "|" << ef << " | " << test_result.recall << "|" 
			<< qps << "|"
			<< total_stats.mean << "|" << ln_stats.mean << "|" << l0_stats.mean << "|"
			<< total_stats.std << "|" << ln_stats.std << "|" << l0_stats.std << "|"
			<< total_stats.three_nines << "|" << ln_stats.three_nines << "|" << l0_stats.three_nines << "|"
			<< total_stats.min << "|" << ln_stats.min << "|" << l0_stats.min << "|"
			<< total_stats.max << "|" << ln_stats.max << "|" << l0_stats.max << "|"
			<< total_stats.total << "|" << ln_stats.total << "|" << l0_stats.total << "|"
			<< endl;
	    }
	}
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


void sift_test1B(
        std::string &algorithm,
		    int subset_size_milllions,
		    int M, 
		    int efConstruction,
		    std::vector<size_t> efs,
		    int qsize, // number of queries in the file
		    int n_queries, //number of queries to run
		    int k,
		    std::string &path_gt,
		    std::string &path_q,
		    size_t repeats,
		    bool permute
		) {
	
	

    size_t vecsize = subset_size_milllions * 1000000;

    size_t vecdim = 128;
    char path_index[1024];
    char path_l0[1024];
    //char path_gt[1024];
    //char *path_q = "bigann/bigann_query.bvecs";
    char *path_data = "bigann/bigann_base.bvecs";
    sprintf(path_index, "sift1b_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);
    sprintf(path_l0, "sift1b_%dm_ef_%d_M_%d.level0", subset_size_milllions, efConstruction, M);

    //sprintf(path_gt, "bigann/gnd/idx_%dM.ivecs", subset_size_milllions);


    unsigned char *massb = new unsigned char[vecdim];

    cout << "Loading GT: " << path_gt << "\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            cout << "Expected rows with 1000 elements but got " << t 
		    << " at line " << i + 1 << endl;
	    return;
        }
    }
    inputGT.close();
	
    cout << "Loading queries:" << path_q << "\n";
    unsigned char *massQ = new unsigned char[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        inputQ.read((char *) massb, in);
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }

    }
    inputQ.close();


    unsigned char *mass = new unsigned char[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    L2SpaceI l2space(vecdim);

    HierarchicalNSW<int> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        if (algorithm == "hnsw") {
            appr_alg = new HierarchicalNSW<int>(&l2space, path_index, false, 0, path_l0);
        } else if (algorithm == "hm-ann"){
            appr_alg = new HmAnn<int>(&l2space, path_index, false, 0, path_l0);
        }
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "No index  " << path_index << "found\n";
        cout << "Building index:\n";
        if (algorithm == "hnsw") {
            appr_alg = new HierarchicalNSW<int>(&l2space, vecsize, M, efConstruction, path_l0);
        } else if (algorithm == "hm-ann") {
            appr_alg = new HmAnn<int>(&l2space, vecsize, M, efConstruction, path_l0);
        }

        input.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        input.read((char *) massb, in);

        for (int j = 0; j < vecdim; j++) {
            mass[j] = massb[j] * (1.0f);
        }

        appr_alg->addPoint((void *) (massb), (size_t) 0);
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
            unsigned char mass[128];
            int j2=0;
#pragma omp critical
            {

                input.read((char *) &in, 4);
                if (in != 128) {
                    cout << "file error";
                    exit(1);
                }
                input.read((char *) massb, in);
                for (int j = 0; j < vecdim; j++) {
                    mass[j] = massb[j];
                }
                j1++;
                j2=j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) (mass), (size_t) j2);


        }
        input.close();
        if (auto hmann = dynamic_cast<HmAnn<int>*>(appr_alg)) {
            std::vector<size_t> level_sizes;
            cout << "Beginning HM-ANN modifications" << endl;
            //TODO exponential
            auto size = appr_alg->max_elements_;
            while (size > 500) {
                level_sizes.push_back(size);
                size /= 500;
            }
            hmann->hm_ann_promote(level_sizes);
        }
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
	cout << "Hops: hier: " << appr_alg->metric_hops_hier <<  " L0: " << appr_alg->metric_distance_computations_hier << endl;
	cout << " Distances: hier: " << appr_alg->metric_distance_computations_hier << " L0: " << appr_alg->metric_distance_computations_l0 << endl;
        appr_alg->saveIndex(path_index);
    }


    vector<std::priority_queue<std::pair<int, labeltype >>> answers;
    cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, l2space, vecdim, answers, k);
    cout << "Loaded gt\n";
    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, qsize, vecsize, n_queries, *appr_alg, vecdim, answers, k, efs, repeats, permute);
	cout << "Hops: hier: " << appr_alg->metric_hops_hier <<  " L0: " << appr_alg->metric_distance_computations_hier << endl;
	cout << " Distances: hier: " << appr_alg->metric_distance_computations_hier << " L0: " << appr_alg->metric_distance_computations_l0 << endl;
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return;


}
