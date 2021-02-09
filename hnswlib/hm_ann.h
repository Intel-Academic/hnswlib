//
// Created by Bryn Keller on 2/8/21.
//

#ifndef HNSW_LIB_HM_ANN_H
#define HNSW_LIB_HM_ANN_H

#include "hnswalg.h"

namespace hnswlib {
    template<typename dist_t>
    class HmAnn : public HierarchicalNSW<dist_t> {

    public:
        HmAnn(SpaceInterface<dist_t> *s) : HierarchicalNSW<dist_t>(s) {

        }

        HmAnn(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false,
              size_t max_elements = 0, std::string level0_path = "hnswlib.level0") :
                HierarchicalNSW<dist_t>(s) {
            this->loadIndex(location, s, max_elements, level0_path);
        }

        HmAnn(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200,
              std::string level0_path = "hnswlib.level0",
              size_t random_seed = 100) : HierarchicalNSW<dist_t>(s, max_elements, M,
                                                          ef_construction, level0_path, random_seed) {
        }

        virtual size_t get_m(size_t level) {
            if (level < 2) {
                return this->M_ * 2;
            }
            return this->M_;
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
            for (; it < indices.cend(); it++) {
                auto v = (*it);
//                std::cout << "Placing " << v << std::endl;
                auto data_point = this->getDataByInternalId(v);
                for (auto i = level_sizes.size(); i >= 1; i--) {
                    if (level_sizes[i - 1] == 0) {
                        if (!flags[i - 1]) {
                            std::cout << "Filled level " << i << std::endl;
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
                            this->mutuallyConnectNewElement(data_point, v, w, i, true);
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
