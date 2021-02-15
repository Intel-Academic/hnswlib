//
// Created by Bryn Keller on 2/8/21.
//

#ifndef HNSW_LIB_HM_ANN_H
#define HNSW_LIB_HM_ANN_H

#include "hnswalg.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <set>

namespace hnswlib {

    struct Request {
        tableint key;
        void *data;

        Request(tableint key, void *data) : key(key), data(data) {}

        Request() : Request(std::numeric_limits<tableint>::max(), nullptr) {}
    };

    struct CacheEntry {
        tableint key;
        void *data;

        CacheEntry() : CacheEntry(std::numeric_limits<tableint>::max(), nullptr) {}

        CacheEntry(tableint key, void *data) : key(key), data(data) {}

        void *get(tableint k) {
            if (k == key)
                return data;
            return nullptr;
        }
    };

    class Prefetcher {
        CacheEntry *cache;
        size_t *jump_table;
        std::queue<Request> requests;
        mutable std::mutex q_mut;
        std::condition_variable cond;
        size_t next;
        std::vector<std::thread> prefetch_threads;
        std::atomic<bool> run;
        size_t data_size;
        std::atomic<size_t> hits;
        std::atomic<size_t> misses;
        size_t cache_len;
        size_t table_len;
        size_t threads_len;
    public:
        Prefetcher(size_t cache_bytes, size_t max_elements, size_t elem_size) :
                data_size(elem_size),
                cache_len(cache_bytes / (elem_size + sizeof(CacheEntry))),
                table_len(max_elements),
                threads_len(1),
                run(true) {
            jump_table = new size_t[table_len];
            for (int i = 0; i < table_len; ++i) {
                jump_table[i] = std::numeric_limits<size_t>::max();
            }
            cache = new CacheEntry[cache_len];
            for (int i = 0; i < cache_len; ++i) {
                cache[i].data = (void*)new char[elem_size];
            }
            std::cout << "Prefetcher has cache for " << cache_len << " elements" << std::endl;
            for (int i = 0; i < threads_len; ++i) {
                prefetch_threads.push_back(std::thread([this]{fetch();}));
            }
        }

        ~Prefetcher() {
            stop();
            join();
            delete[] cache;
            delete[] jump_table;
        }

        void report() {
            auto h = static_cast<float>(hits);
            auto m = static_cast<float>(misses);
            auto total = h + m;
            std::cout << "Overall prefetcher cache hits: " << hits << " misses: " << misses << " rate: "
                      << (h/total) << std::endl;
        }

        void prefetch(tableint key, void *address) {
            {
                std::lock_guard<std::mutex> lock(q_mut);
                requests.emplace(key, address);
            }
            cond.notify_one();
        }

        void *get(tableint key) {
            auto loc = jump_table[key];
            if (loc != std::numeric_limits<size_t>::max()) {
                CacheEntry &entry = cache[loc];
                void *ptr = entry.get(key);
                if (ptr != nullptr) {
                    hits++;
                    return ptr;
                }
            }
            misses++;
            return nullptr;
        }

        void stop() {
            run = false;
        }

        void join() {
            for (auto &t: prefetch_threads) {
                t.join();
            }
        }

        void fetch() {
            std::vector<Request> batch;
            batch.reserve(2000);
            while (run) {
                {
                    std::unique_lock<std::mutex> lock(q_mut);
                    while(requests.empty()) {
                        cond.wait(lock);
                    }
                    while (!requests.empty()) {
                        auto req = requests.front();
                        batch.push_back(req);
                        requests.pop();
                    }
                }
                for (auto req: batch) {
                    //Check to see if it's already cached
                    auto check = jump_table[req.key];
                    if (check != std::numeric_limits<size_t>::max()) {
                        CacheEntry &check_entry = cache[check];
                        if (check_entry.get(req.key) != nullptr) {
                            //still in cache
                            continue;
                        }
                    }
                    CacheEntry &entry = cache[next];
                    entry.key = req.key; // Invalidates the cache entry for any previous key
                    memcpy(entry.data, req.data, data_size); // Cache
                    jump_table[req.key] = next; // Now we can find the cache
                    next++; // Move the insertion point forward
                    next = next % cache_len; // or maybe wrap around
                }
                batch.clear();
            }
        }
    };

    const size_t CACHE_BYTES =1024 * 1024 * 1024;

    template<typename dist_t>
    class HmAnn : public HierarchicalNSW<dist_t> {
        mutable Prefetcher* prefetcher_;

    public:
        HmAnn(SpaceInterface<dist_t> *s) : HierarchicalNSW<dist_t>(s), prefetcher_(nullptr) {

        }

        HmAnn(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false,
              size_t max_elements = 0, std::string level0_path = "hnswlib.level0") :
                HierarchicalNSW<dist_t>(s){
            this->loadIndex(location, s, max_elements, level0_path);
            prefetcher_ = new Prefetcher(CACHE_BYTES, this->max_elements_, this->data_size_);
        }

        HmAnn(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200,
              std::string level0_path = "hnswlib.level0",
              size_t random_seed = 100) : HierarchicalNSW<dist_t>(s, max_elements, M,
                                                                  ef_construction, level0_path, random_seed) {
            prefetcher_ = new Prefetcher(CACHE_BYTES, this->max_elements_, this->size_data_per_element_);
        }

        template<bool use_prefetch> char* getDataWithPrefetch(tableint internal_id) const {
            if (use_prefetch) {
                auto ptr = (char*)prefetcher_->get(internal_id);
                if (nullptr == ptr) {
                    ptr = this->getDataByInternalId(internal_id);
                }
                return ptr;
            } else {
                return this->getDataByInternalId(internal_id);
            }
        }

        template<bool has_deletions, bool collect_metrics = true, bool prefetch = false, bool use_prefetch = false, bool machine_prefetch = true>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                typename HmAnn<dist_t>::CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, size_t level) const {
            VisitedList *vl = this->visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                    typename HmAnn<dist_t>::CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                    typename HmAnn<dist_t>::CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !this->isMarkedDeleted(ep_id)) {
                dist_t dist = this->fstdistfunc_(data_point, this->getDataWithPrefetch<use_prefetch>(ep_id), this->dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) this->get_linklist_at_level(current_node_id, level);
                size_t size = this->getListCount((linklistsizeint *) data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics) {
                    this->metric_hops_l0++;
                    this->metric_distance_computations_l0 += size;
                }

                if (machine_prefetch) {
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(
                        this->data_level0_memory_ + (*(data + 1)) * this->size_data_per_element_ + this->offsetData_,
                        _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif
                }
                if (prefetch) {
                    this->prefetcher_->prefetch(current_node_id, this->getDataByInternalId(current_node_id));
                    for (size_t j = 1; j <= size; j++) {
                        int candidate_id = *(data + j);
                        this->prefetcher_->prefetch(candidate_id, this->getDataByInternalId(candidate_id));
                    }
                }
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
                    if (machine_prefetch) {
#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                        _mm_prefetch(this->data_level0_memory_ + (*(data + j + 1)) * this->size_data_per_element_ +
                                     this->offsetData_,
                                     _MM_HINT_T0);////////////
#endif
                    }
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = this->getDataWithPrefetch<use_prefetch>(candidate_id);
                        dist_t dist = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
                            if (machine_prefetch) {
#ifdef USE_SSE
                                _mm_prefetch(this->data_level0_memory_ +
                                             candidate_set.top().second * this->size_data_per_element_ +
                                             this->offsetLevel0_,///////////
                                             _MM_HINT_T0);////////////////////////
#endif
                            }

                            if (!has_deletions || !this->isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            this->visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }


        QueryResult
        searchKnn(const void *query_data, size_t k) const {

            QueryResult result = {};
            if (this->cur_element_count == 0) return result;
            StopW timer;
            tableint currObj = this->get_best_entry_point(query_data, 2);
            result.times.ln_micros = timer.getElapsedTimeMicro();
            timer.reset();
            //Now process layer 1, prefetching as we go
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                    typename HmAnn<dist_t>::CompareByFirst> top_candidates;
            //TODO: extract this if into a method?
            if (this->has_deletions_) {
                top_candidates = this->searchBaseLayerST<true, true, true, true, true>(
                        currObj, query_data, std::max(this->ef_, k), 1);
            } else {
                top_candidates = this->searchBaseLayerST<false, true, true, true, true>(
                        currObj, query_data, std::max(this->ef_, k), 1);
            }

            //Now use the results from layer 1 as entry points for a multithreaded ef 1 search of L0
            std::vector<tableint> starts;
            starts.reserve(top_candidates.size());

//            while (top_candidates.size() > std::max(this->ef_, k)) {
//                top_candidates.pop();
//            }

            while(!top_candidates.empty()) {
                tableint item = top_candidates.top().second;
                starts.push_back(item);
                top_candidates.pop();
            }
            std::unordered_set<tableint> unique_nodes;
#pragma omp parallel for num_threads(4) //TODO make a param for this
            for (int i = 0; i < starts.size(); ++i) {
                auto ep = starts[i];
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        typename HmAnn<dist_t>::CompareByFirst> candidates;
                if (this->has_deletions_) {
                    candidates = this->searchBaseLayerST<true, true, false, true, true>(
                            ep, query_data, 2, 0);
                } else {
                    candidates = this->searchBaseLayerST<false, true, false, true, true>(
                            ep, query_data, 2, 0);
                }
#pragma omp critical
                {
                    while(!candidates.empty()) {
                        auto res = candidates.top();
                        // Each search might return duplicates of other searches, we only
                        // want them in the result list once
                        if (0 == unique_nodes.count(res.second)) {
                            unique_nodes.insert(res.second);
                            top_candidates.push(res);
                        }
                        candidates.pop();
                    }
                }
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.result.push(std::pair<dist_t, labeltype>(rez.first, this->getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            result.times.l0_micros = timer.getElapsedTimeMicro();
            static int calls = 0;
            calls++;
            if (calls % 10000 == 0)
                prefetcher_->report();
            return result;
        }

        void hm_ann_promote(std::vector<size_t> level_sizes) {
            if (level_sizes.empty()) {
                std::cerr << "WARNING: no level sizes, database will have no hierarchy" << std::endl;
            }
            std::cout << "Promoting with levels ";
            std::copy(level_sizes.begin(), level_sizes.end(),
                      std::ostream_iterator<size_t>(std::cout, " "));
            std::cout << std::endl;
            std::vector<size_t> degrees;
            auto element_count = this->cur_element_count;
            degrees.reserve(element_count);
            for (auto i = 0; i < element_count; i++) {
                auto size = this->getListCount(this->get_linklist0(i));
                degrees.push_back(size);
            }
            std::vector<size_t> indices(degrees.size());
            std::iota(indices.begin(), indices.end(), 0);
            //Sort indices by degree of node at that index, descending
            std::stable_sort(indices.begin(), indices.end(),
                             [&degrees](size_t l, size_t r) { return degrees[l] > degrees[r]; });

            //Clear old level info
            this->free_lists();
            this->init_lists();
            for (int i = 0; i < this->max_elements_; ++i) {
                this->element_levels_[i] = 0;
            }
            //Put the entry point in the highest level
            auto entry_point = indices[0];
            this->maxlevel_ = level_sizes.size();
            this->set_level(entry_point, this->maxlevel_);
            this->enterpoint_node_ = entry_point;
            auto it = indices.cbegin();
            it++;
            std::vector flags(level_sizes.size(), false);
#pragma omp parallel for
            for (; it < indices.cend(); it++) {
                auto v = (*it);
//                std::cout << "Placing " << v << std::endl;
                // Take update lock to prevent race conditions on an element with insertion/update at the same time.
                std::unique_lock<std::mutex> lock_el_update(
                        this->link_list_update_locks_[(v & (this->max_update_element_locks - 1))]);
                std::unique_lock<std::mutex> lock_el(this->link_list_locks_[v]);

                auto data_point = this->getDataByInternalId(v);
                entry_point = this->enterpoint_node_;
                for (auto i = level_sizes.size(); i >= 1; i--) {
                    if (level_sizes[i - 1] == 0) {
                        if (!flags[i - 1]) {
                            std::cout << "Filled level " << i << std::endl;
//                            this->checkIntegrity();
                            flags[i - 1] = true;
                        }
                        auto[closest, dist] = this->find_closest_neighbor(entry_point, data_point, i);
                        entry_point = closest;
                    } else {
                        this->set_level(v, i);
//                        std::cout << "Placing " << v << " in level " << i << std::endl;
                        for (auto j = i; j >= 1; j--) {
                            auto w = this->searchBaseLayer(entry_point, data_point, j);
                            std::vector<std::pair<dist_t, tableint>> vec;
                            while (!w.empty()) {
                                //Filter out self-references, TODO why is this even happening?
                                auto top = w.top();
                                w.pop();
                                if (top.second != v) {
                                    vec.push_back(top);
                                }
                            }
                            for (auto it = vec.begin(); it < vec.end(); it++) {
                                w.push(*it);
                            }
                            entry_point = this->mutuallyConnectNewElement(data_point, v, w, j, true);
                            auto[ns, sz] = this->get_neighbors(v, j);
                            if (sz == 0) {
                                std::cerr << "No neighbors added for " << v << " in level " << j << std::endl;
                                throw std::runtime_error("No neighbors added");
                            }
                            //TODO: shrink connections?
                            level_sizes[j - 1]--;
                        }
                        break;
                    }
                }
            }
        }
    };
}
#endif //HNSW_LIB_HM_ANN_H
