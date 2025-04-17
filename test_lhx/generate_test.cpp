#include "common.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " [dim] [max_elements] [num_threads]"
              << std::endl;
    return 0;
  }
  int dim = atoi(argv[1]);
  int max_elements = atoi(argv[2]);
  int num_threads = atoi(argv[3]);

  std::string index_file_name = "hnsw_index." + std::to_string(dim) + "_" +
                                std::to_string(max_elements) + "_" +
                                std::to_string(num_threads);

  uint64_t cost = 0;
  auto hnsw = GenerateHnswGraph(dim, max_elements, num_threads, cost);
  std::cout << "BUILD: total_cost=" << cost
            << "ms, avg=" << (cost * 1.0 / max_elements) << "ms" << std::endl;

  Timer timer;
  hnsw->saveIndex(index_file_name);
  std::cout << "SAVE: total_cost=" << timer.ElapsedMicroseconds() << "ms"
            << std::endl;

  return 0;
}