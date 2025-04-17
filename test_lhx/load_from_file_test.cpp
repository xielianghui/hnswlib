#include "common.h"

struct CompareByFirst {
  constexpr bool
  operator()(std::pair<double, hnswlib::labeltype> const &a,
             std::pair<double, hnswlib::labeltype> const &b) const noexcept {
    return a.first < b.first;
  }
};

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [index_file_path] [ef] [topk]"
              << std::endl;
    return 0;
  }

  // load from file
  std::string filename(argv[1]);
  size_t pos = filename.find('.');
  if (pos == std::string::npos) {
    std::cerr << "Invalid filename format" << std::endl;
    return 1;
  }
  std::string numbers_part = filename.substr(pos + 1);
  std::stringstream ss(numbers_part);
  std::string token;
  std::vector<int> numbers;
  while (std::getline(ss, token, '_')) {
    numbers.push_back(std::stoi(token));
  }
  if (numbers.size() != 5) {
    std::cerr << "Unexpected number of elements" << std::endl;
    return 1;
  }
  int dim = numbers[0];
  int max_elements = numbers[1];
  size_t m = numbers[2];
  size_t ef_construction = numbers[3];
  int num_threads = numbers[4];

  Timer timer;
  hnswlib::L2Space space(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw(
      new hnswlib::HierarchicalNSW<float>(&space, filename));
  std::cout << "FILE LOAD: total_cost=" << timer.ElapsedMicroseconds()
            << " μs, mem_usage=" << GetMemoryUsageMB() << " MB" << std::endl;

  if (argc == 2) {
    return 0;
  }

  size_t ef = 10;
  if (argc >= 3) {
    alg_hnsw->setEf(atoi(argv[2]));
  }

  size_t topk = 10;
  if (argc >= 4) {
    topk = atoi(argv[3]);
  }

  static const size_t kQueryCount = 10;
  uint64_t cost = 0;
  uint64_t bruteforce_cost = 0;
  double recall = 0.0;
  std::mt19937 rng;
  rng.seed(static_cast<unsigned int>(std::time(0)));
  std::uniform_int_distribution<int> distrib_int(1, max_elements);

  for (size_t i = 0; i < kQueryCount; i++) {
    char *query_data = alg_hnsw->getDataByInternalId(distrib_int(rng) - 1);

    // query use hnsw
    timer.Reset();
    auto res = alg_hnsw->searchKnnCloserFirst(query_data, topk);
    cost += timer.ElapsedMicroseconds();

    // query use brute force
    timer.Reset();
    std::unordered_set<hnswlib::labeltype> knns;
    std::priority_queue<std::pair<double, hnswlib::labeltype>,
                        std::vector<std::pair<double, hnswlib::labeltype>>,
                        CompareByFirst>
        candidates;
    for (size_t j = 0; j < max_elements; j++) {
      double dist =
          alg_hnsw->fstdistfunc_(query_data, alg_hnsw->getDataByInternalId(j),
                                 alg_hnsw->dist_func_param_);
      candidates.push(std::make_pair(dist, alg_hnsw->getExternalLabel(j)));
      if (candidates.size() > topk) {
        candidates.pop();
      }
    }
    assert(candidates.size() == topk);
    while (!candidates.empty()) {
      knns.insert(candidates.top().second);
      candidates.pop();
    }
    bruteforce_cost += timer.ElapsedMicroseconds();

    // calc recall
    size_t correct_count = 0;
    for (auto &pair : res) {
      if (knns.count(pair.second) > 0) {
        ++correct_count;
      }
    }
    recall += (correct_count * 1.0) / topk;
  }
  std::cout << "QUERY: total_cost=" << cost
            << " μs, avg_cost=" << (cost / kQueryCount)
            << " μs, bruteforce_cost=" << bruteforce_cost
            << " μs, avg_bruteforce_cost=" << (bruteforce_cost / kQueryCount)
            << " μs, recall=" << (recall / kQueryCount) << std::endl;

  return 0;
}