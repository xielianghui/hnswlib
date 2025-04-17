#include "common.h"

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static std::string kMetaFileName = "vem";
static std::string kLinkListFileName = "vex";
static std::string kLevelData0FileName = "vec";

static void *linlist_map = nullptr;
static size_t linlist_size = 0;
static void *level0_data_map = nullptr;
static size_t level0_data_size = 0;

static void
Save2File(std::unique_ptr<hnswlib::HierarchicalNSW<float>> &alg_hnsw) {
  Timer timer;
  std::vector<std::thread> threads;

  // 1. save meta value
  threads.emplace_back(std::thread([&alg_hnsw]() {
    std::ofstream output(kMetaFileName, std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("Failed to open file: " + kMetaFileName);
    }
    auto meta_value = GetHnswMetaValue(alg_hnsw);
    auto meta_str = meta_value.Serialize();
    output.write(meta_str.c_str(), meta_str.size());
    output.close();
  }));

  // 2. save linklist
  threads.emplace_back(std::thread([&alg_hnsw]() {
    std::ofstream output(kLinkListFileName, std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("Failed to open file: " + kLinkListFileName);
    }
    for (int i = 0; i < alg_hnsw->cur_element_count; i++) {
      if (alg_hnsw->element_levels_[i] > 0) {
        unsigned int linkListSize =
            sizeof(int) +
            alg_hnsw->size_links_per_element_ * alg_hnsw->element_levels_[i];
        writeBinaryPOD(output, linkListSize);
        writeBinaryPOD(output, i);
        if (linkListSize)
          output.write(alg_hnsw->linkLists_[i], linkListSize);
      }
    }
    output.close();
  }));

  // 3. save level0 data
  threads.emplace_back(std::thread([&alg_hnsw]() {
    std::ofstream output(kLevelData0FileName, std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("Failed to open file: " + kLinkListFileName);
    }
    output.write(alg_hnsw->data_level0_memory_,
                 alg_hnsw->cur_element_count *
                     alg_hnsw->size_data_per_element_);
    output.close();
  }));

  for (auto &thread : threads) {
    thread.join();
  }
  std::cout << "DB SAVE: total_cost=" << timer.Elapsed() << " ms" << std::endl;
}

static std::pair<std::unique_ptr<hnswlib::HierarchicalNSW<float>>,
                 std::unique_ptr<hnswlib::L2Space>>
LoadFromFile() {
  Timer timer;
  Timer phase_timer;
  // 1. load meta value
  std::ifstream input(kMetaFileName, std::ios::binary);
  if (!input.is_open()) {
    std::cout << "Failed to open file: " << kMetaFileName << std::endl;
  }
  std::ostringstream ss;
  ss << input.rdbuf();
  std::string meta_value_str = ss.str();
  HnswMetaValue meta_value;
  meta_value.Deserialize(meta_value_str);

  std::unique_ptr<hnswlib::L2Space> space(
      new hnswlib::L2Space(meta_value.dim_));
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw(
      new hnswlib::HierarchicalNSW<float>(space.get()));

  // load meta value
  alg_hnsw->offsetLevel0_ = meta_value.offsetLevel0_;
  alg_hnsw->max_elements_ = meta_value.max_elements_;
  alg_hnsw->cur_element_count = meta_value.cur_element_count;
  alg_hnsw->size_data_per_element_ = meta_value.size_data_per_element_;
  alg_hnsw->label_offset_ = meta_value.label_offset_;
  alg_hnsw->offsetData_ = meta_value.offsetData_;
  alg_hnsw->maxlevel_ = meta_value.maxlevel_;
  alg_hnsw->enterpoint_node_ = meta_value.enterpoint_node_;
  alg_hnsw->maxM_ = meta_value.maxM_;
  alg_hnsw->maxM0_ = meta_value.maxM0_;
  alg_hnsw->M_ = meta_value.M_;
  alg_hnsw->mult_ = meta_value.mult_;
  alg_hnsw->ef_construction_ = meta_value.ef_construction_;

  alg_hnsw->data_size_ = space->get_data_size();
  alg_hnsw->fstdistfunc_ = space->get_dist_func();
  alg_hnsw->dist_func_param_ = space->get_dist_func_param();

  alg_hnsw->size_links_per_element_ =
      alg_hnsw->maxM_ * sizeof(hnswlib::tableint) +
      sizeof(hnswlib::linklistsizeint);
  alg_hnsw->size_links_level0_ = alg_hnsw->maxM0_ * sizeof(hnswlib::tableint) +
                                 sizeof(hnswlib::linklistsizeint);

  alg_hnsw->element_levels_.resize(alg_hnsw->max_elements_, 0);
  alg_hnsw->revSize_ = 1.0 / alg_hnsw->mult_;
  alg_hnsw->ef_ = 10;

  std::cout << "DB LOAD: load_meta_value_cost=" << phase_timer.Elapsed()
            << " ms" << std::endl;

  std::vector<std::thread> threads;
  // malloc locks
  threads.emplace_back(std::thread([&alg_hnsw]() {
    Timer timer;
    std::vector<std::mutex>(alg_hnsw->MAX_LABEL_OPERATION_LOCKS)
        .swap(alg_hnsw->label_op_locks_);
    std::vector<std::mutex>(alg_hnsw->max_elements_)
        .swap(alg_hnsw->link_list_locks_);
    alg_hnsw->visited_list_pool_.reset(
        new hnswlib::VisitedListPool(1, alg_hnsw->max_elements_));
    std::cout << "DB LOAD: malloc_locks_cost=" << timer.Elapsed() << " ms"
              << std::endl;
  }));

  threads.emplace_back(std::thread([&alg_hnsw]() {
    Timer timer;
    // load link list
    alg_hnsw->linkLists_ =
        (char **)malloc(sizeof(void *) * alg_hnsw->max_elements_);
    if (alg_hnsw->linkLists_ == nullptr) {
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate linklists");
    }

    int fd = open(kLinkListFileName.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file: " + kLinkListFileName);
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      throw std::runtime_error("Failed to stat file: " + kLinkListFileName);
    }
    linlist_size = sb.st_size;
    linlist_map = mmap(0, linlist_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (linlist_map == MAP_FAILED) {
      throw std::runtime_error("Failed to map file: " + kLinkListFileName);
    }

    char *start = (char *)linlist_map;
    char *data = (char *)linlist_map;
    size_t gap = 0;
    while ((data - start) < linlist_size) {
      unsigned int linkListSize = *(unsigned int *)data;
      data += sizeof(unsigned int);
      int id = *(unsigned int *)data;
      data += sizeof(unsigned int);

      alg_hnsw->element_levels_[id] =
          linkListSize / alg_hnsw->size_links_per_element_;
      alg_hnsw->linkLists_[id] = data;
      data += linkListSize;
    }
    close(fd);
    std::cout << "DB LOAD: load_linklist_cost=" << timer.Elapsed() << " ms"
              << std::endl;
  }));

  threads.emplace_back(std::thread([&alg_hnsw]() {
    // load data level0
    // alg_hnsw->data_level0_memory_ = (char *)malloc(
    //     alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
    // if (alg_hnsw->data_level0_memory_ == nullptr) {
    //   throw std::runtime_error(
    //       "Not enough memory: loadIndex failed to allocate level0");
    // }
    Timer timer;
    int fd = open(kLevelData0FileName.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file: " + kLevelData0FileName);
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      throw std::runtime_error("Failed to stat file: " + kLevelData0FileName);
    }
    level0_data_size = sb.st_size;
    level0_data_map = mmap(0, level0_data_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (level0_data_map == MAP_FAILED) {
      throw std::runtime_error("Failed to map file: " + kLevelData0FileName);
    }

    alg_hnsw->data_level0_memory_ = (char *)level0_data_map;
    close(fd);
    std::cout << "DB LOAD: load_level0_cost=" << timer.Elapsed() << " ms"
              << std::endl;
  }));

  for (auto &thread : threads) {
    thread.join();
  }

  std::cout << "DB LOAD: total_cost=" << timer.Elapsed() << " ms" << std::endl;

  // only used for update
  for (size_t i = 0; i < alg_hnsw->max_elements_; ++i) {
    alg_hnsw->label_lookup_[alg_hnsw->getExternalLabel(i)] = i;
  }

  return std::make_pair(std::move(alg_hnsw), std::move(space));
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [index_file_path]" << std::endl;
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

  Save2File(alg_hnsw);

  auto ret = LoadFromFile();
  auto &reload_alg_hnsw = ret.first;

  // check valid
  assert(GetHnswMetaValue(alg_hnsw) == GetHnswMetaValue(reload_alg_hnsw));
  std::string query_data = GetRandomVectorData(dim);
  auto top100 = alg_hnsw->searchKnnCloserFirst((void *)query_data.data(), 100);
  auto reload_top100 =
      reload_alg_hnsw->searchKnnCloserFirst((void *)query_data.data(), 100);
  assert(top100 == reload_top100);

  // query test
  reload_alg_hnsw->setEf(2000);
  int topk = 20;
  static const size_t kQueryCount = 10;
  uint64_t cost = 0;
  uint64_t bruteforce_cost = 0;
  double recall = 0.0;
  std::mt19937 rng;
  rng.seed(static_cast<unsigned int>(std::time(0)));
  std::uniform_int_distribution<int> distrib_int(1, max_elements);

  for (size_t i = 0; i < kQueryCount; i++) {
    char *query_data =
        reload_alg_hnsw->getDataByInternalId(distrib_int(rng) - 1);

    // query use hnsw
    timer.Reset();
    auto res = reload_alg_hnsw->searchKnnCloserFirst(query_data, topk);
    cost += timer.ElapsedMicroseconds();
  }
  std::cout << "QUERY: total_cost=" << cost
            << " μs, avg_cost=" << (cost / kQueryCount) << std::endl;

  if (munmap(level0_data_map, level0_data_size) == -1) {
    throw std::runtime_error("Failed to unmap file: " + kLevelData0FileName);
  }
  reload_alg_hnsw->data_level0_memory_ = nullptr;
  if (munmap(linlist_map, linlist_size) == -1) {
    throw std::runtime_error("Failed to unmap file: " + kLinkListFileName);
  }
  for (size_t i = 0; i < reload_alg_hnsw->max_elements_; i++) {
    reload_alg_hnsw->linkLists_[i] = nullptr;
  }
  return 0;
}