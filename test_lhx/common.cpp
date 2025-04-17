#include "./common.h"

static const int KM = 16;
static const int KEFConstruction = 200;

std::unique_ptr<hnswlib::HierarchicalNSW<float>>
GenerateHnswGraph(const int dim, const int max_elements, const int num_threads,
                  uint64_t &cost) {
  // init index
  hnswlib::L2Space space(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw(
      new hnswlib::HierarchicalNSW<float>(&space, max_elements, KM,
                                          KEFConstruction));

  // generate random data
  std::mt19937 rng;
  rng.seed(47);
  std::uniform_real_distribution<float> distrib_real(1, 100);
  size_t data_point_size = space.get_data_size();
  std::string data;
  data.resize(data_point_size * max_elements);
  for (int i = 0; i < max_elements; i++) {
    char *point_data = const_cast<char *>(data.data()) + i * data_point_size;
    for (int j = 0; j < dim; j++) {
      char *vec_data = point_data + j * sizeof(float);
      float value = distrib_real(rng);
      *(float *)vec_data = value;
    }
  }

  // add data to index
  Timer timer;
  ParallelFor(0, max_elements, num_threads, [&](size_t i, size_t thread_id) {
    alg_hnsw->addPoint(
        (void *)(const_cast<char *>(data.data()) + data_point_size * i), i);
  });
  cost = timer.Elapsed();
  return alg_hnsw;
}