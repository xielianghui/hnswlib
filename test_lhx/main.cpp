#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>

#include "leveldb/db.h"
#include "leveldb/env.h"
#include "leveldb/options.h"

#include "../hnswlib/hnswlib.h"

static const int Km = 16;
static const int Kef_construction = 200;
static const int kquery_size = 5;
static const std::string Kdbname = "./index_db";

static int dim = 0;
static int max_elements = 0;
static int topk = 5;

static std::string DataToString(char *data) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < dim; ++i) {
    ss << *((float *)(data) + i);
    if (i != dim - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

static void DoQuery(const hnswlib::HierarchicalNSW<float> *alg_hnsw,
                    const std::string &source_data,
                    const size_t data_point_size,
                    const std::vector<std::string> &query_datas) {
  char *data = const_cast<char *>(source_data.data());
  for (int i = 0; i < query_datas.size(); i++) {
    char *query_data = const_cast<char *>(query_datas[i].data());
    std::cout << "Query #" << i << ", " << DataToString(query_data)
              << std::endl;

    auto result = alg_hnsw->searchKnnCloserFirst(query_data, topk);
    std::cout << "Result #" << i << ",";
    for (auto &it : result) {
      std::cout << DataToString((data + it.second * data_point_size)) << ", ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " [dim] [max_elements]" << std::endl;
    return 0;
  }

  dim = atoi(argv[1]);
  max_elements = atoi(argv[2]);
  if (max_elements < topk) {
    topk = max_elements;
  }

  // open leveldb
  leveldb::DestroyDB(Kdbname, leveldb::Options());

  leveldb::DB *db = nullptr;
  leveldb::Options opts;
  opts.create_if_missing = true;
  auto s = leveldb::DB::Open(opts, Kdbname, &db);
  if (!s.ok()) {
    std::cout << "Failed to open db, " << s.ToString() << std::endl;
  }
  std::unique_ptr<leveldb::DB> db_deleter(db);

  // init index
  hnswlib::L2Space space(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw(
      new hnswlib::HierarchicalNSW<float>(&space, max_elements, Km,
                                          Kef_construction));

  // generate random data
  std::mt19937 rng;
  rng.seed(47);
  std::uniform_real_distribution<float> distrib_real(1, 100);

  size_t data_point_size = space.get_data_size();
  std::string source_data;
  source_data.resize(data_point_size * max_elements);
  for (int i = 0; i < max_elements; i++) {
    char *point_data =
        const_cast<char *>(source_data.data()) + i * data_point_size;
    for (int j = 0; j < dim; j++) {
      char *vec_data = point_data + j * sizeof(float);
      float value = distrib_real(rng);
      *(float *)vec_data = value;
    }
  }

  // add data to index
  for (int i = 0; i < max_elements; i++) {
    hnswlib::labeltype label = i;
    char *point_data =
        const_cast<char *>(source_data.data()) + i * data_point_size;
    alg_hnsw->addPoint(point_data, label);
  }

  // generate query random vectors
  std::vector<std::string> query_datas;
  query_datas.resize(kquery_size);
  for (int i = 0; i < kquery_size; i++) {
    query_datas[i].resize(data_point_size);
    for (int j = 0; j < dim; j++) {
      size_t offset = j * sizeof(float);
      char *vec_data = const_cast<char *>(query_datas[i].data()) + offset;
      float value = distrib_real(rng);
      *(float *)vec_data = value;
    }
  }

  DoQuery(alg_hnsw.get(), source_data, data_point_size, query_datas);

  // save in db

  return 0;
}
