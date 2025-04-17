#include "common.h"

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " [dim] [max_elements] [m] [ef_construction] [num_threads]"
              << std::endl;
    return 0;
  }
  int dim = atoi(argv[1]);
  int max_elements = atoi(argv[2]);
  int m = atoi(argv[3]);
  int ef_construction = atoi(argv[4]);
  int num_threads = atoi(argv[5]);

  std::string index_file_name =
      "hnsw_index." + std::to_string(dim) + "_" + std::to_string(max_elements) +
      "_" + std::to_string(m) + "_" + std::to_string(ef_construction) + "_" +
      std::to_string(num_threads);

  uint64_t cost = 0;
  auto pair = GenerateHnswGraph(dim, max_elements, m, ef_construction,
                                num_threads, cost);
  std::cout << "BUILD: total_cost=" << cost
            << " ms, avg=" << (cost * 1.0 / max_elements) << " ms" << std::endl;

  Timer timer;
  pair.first->saveIndex(index_file_name);
  std::cout << "SAVE: total_cost=" << timer.ElapsedMicroseconds() << " μs"
            << std::endl;

  return 0;
}