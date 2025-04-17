#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <thread>

#include "../hnswlib/hnswlib.h"

class Timer {
public:
  Timer() : startTimePoint(std::chrono::high_resolution_clock::now()) {}

  void Reset() { startTimePoint = std::chrono::high_resolution_clock::now(); }

  // 默认返回毫秒数
  uint64_t Elapsed() const { return ElapsedMilliseconds(); }

  // 返回毫秒数
  uint64_t ElapsedMilliseconds() const {
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTimePoint - startTimePoint);
    return static_cast<uint64_t>(elapsedTime.count());
  }

  // 返回微秒数
  uint64_t ElapsedMicroseconds() const {
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(
        endTimePoint - startTimePoint);
    return static_cast<uint64_t>(elapsedTime.count());
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
};

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn) {
  if (numThreads <= 0) {
    numThreads = std::thread::hardware_concurrency();
  }

  if (numThreads == 1) {
    for (size_t id = start; id < end; id++) {
      fn(id, 0);
    }
  } else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);

          if (id >= end) {
            break;
          }

          try {
            fn(id, threadId);
          } catch (...) {
            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
    if (lastException) {
      std::rethrow_exception(lastException);
    }
  }
}

std::pair<std::unique_ptr<hnswlib::HierarchicalNSW<float>>,
          std::unique_ptr<hnswlib::L2Space>>
GenerateHnswGraph(const int dim, const int max_elements, size_t m,
                  size_t ef_construction, const int num_threads,
                  uint64_t &cost);

double GetMemoryUsageMB();

enum HnswInnerKeyType : uint8_t {
  kMeta = 1,
  kLinkList = 2,
  kLevel0Data = 3,
};

std::string EncodeHnswMetaKey(uint16_t table_id, uint64_t uin, uint8_t field_id,
                              uint16_t sub_field_id);

std::string EncodeHnswLinkListKey(uint16_t table_id, uint64_t uin,
                                  uint8_t field_id, uint16_t sub_field_id,
                                  int internal_id);

std::string EncodeHnswLinkListBeginKey(uint16_t table_id, uint64_t uin,
                                       uint8_t field_id, uint16_t sub_field_id);

std::string EncodeHnswLinkListEndKey(uint16_t table_id, uint64_t uin,
                                     uint8_t field_id, uint16_t sub_field_id);

std::string EncodeHnswLevel0DataKey(uint16_t table_id, uint64_t uin,
                                    uint8_t field_id, uint16_t sub_field_id,
                                    int internal_id);

std::string EncodeHnswLevel0DataBeginKey(uint16_t table_id, uint64_t uin,
                                         uint8_t field_id,
                                         uint16_t sub_field_id);

std::string EncodeHnswLevel0DataEndKey(uint16_t table_id, uint64_t uin,
                                       uint8_t field_id, uint16_t sub_field_id);

template <typename T> void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T> void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

struct HnswMetaValue {
  uint32_t offsetLevel0_ = 0;
  uint32_t max_elements_ = 0;
  uint32_t cur_element_count = 0;
  uint32_t size_data_per_element_ = 0;
  uint32_t label_offset_ = 0;
  uint32_t offsetData_ = 0;
  int maxlevel_ = 0;
  int enterpoint_node_ = 0;
  uint32_t maxM_ = 0;
  uint32_t maxM0_ = 0;
  uint32_t M_ = 0;
  double mult_ = 0.0;
  uint32_t ef_construction_ = 0;

  int dim_ = 0;

  std::string Serialize() const {
    std::stringstream output;
    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    writeBinaryPOD(output, dim_);
    return output.str();
  }

  void Deserialize(const std::string &value) const {
    std::stringstream input(value);
    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    readBinaryPOD(input, dim_);
  }

  bool operator==(const HnswMetaValue &other) const {
    return offsetLevel0_ == other.offsetLevel0_ &&
           max_elements_ == other.max_elements_ &&
           cur_element_count == other.cur_element_count &&
           size_data_per_element_ == other.size_data_per_element_ &&
           label_offset_ == other.label_offset_ &&
           offsetData_ == other.offsetData_ && maxlevel_ == other.maxlevel_ &&
           enterpoint_node_ == other.enterpoint_node_ && maxM_ == other.maxM_ &&
           maxM0_ == other.maxM0_ && M_ == other.M_ && mult_ == other.mult_ &&
           ef_construction_ == other.ef_construction_ && dim_ == other.dim_;
  }
};

inline HnswMetaValue
GetHnswMetaValue(std::unique_ptr<hnswlib::HierarchicalNSW<float>> &alg_hnsw) {
  HnswMetaValue value;
  value.offsetLevel0_ = alg_hnsw->offsetLevel0_;
  value.max_elements_ = alg_hnsw->max_elements_;
  value.cur_element_count = alg_hnsw->cur_element_count;
  value.size_data_per_element_ = alg_hnsw->size_data_per_element_;
  value.label_offset_ = alg_hnsw->label_offset_;
  value.offsetData_ = alg_hnsw->offsetData_;
  value.maxlevel_ = alg_hnsw->maxlevel_;
  value.enterpoint_node_ = alg_hnsw->enterpoint_node_;
  value.maxM_ = alg_hnsw->maxM_;
  value.maxM0_ = alg_hnsw->maxM0_;
  value.M_ = alg_hnsw->M_;
  value.mult_ = alg_hnsw->mult_;
  value.ef_construction_ = alg_hnsw->ef_construction_;

  value.dim_ = alg_hnsw->data_size_ / sizeof(float);
  return value;
}

inline void
GetHnswLinkListData(std::unique_ptr<hnswlib::HierarchicalNSW<float>> &alg_hnsw,
                    int internal_id, char *&data, size_t &data_size) {
  data_size = alg_hnsw->element_levels_[internal_id] > 0
                  ? alg_hnsw->size_links_per_element_ *
                        alg_hnsw->element_levels_[internal_id]
                  : 0;
  data = (char *)alg_hnsw->linkLists_[internal_id];
}

inline void
GetHnswLevel0Data(std::unique_ptr<hnswlib::HierarchicalNSW<float>> &alg_hnsw,
                  int internal_id, char *&data, size_t &data_size) {
  data_size = alg_hnsw->size_data_per_element_;
  data = alg_hnsw->data_level0_memory_ +
         internal_id * alg_hnsw->size_data_per_element_ +
         alg_hnsw->offsetLevel0_;
}

inline std::string GetRandomVectorData(const int dim) {
  std::string vector_data;
  vector_data.resize(dim * sizeof(float));
  std::mt19937 rng;
  rng.seed(std::time(0));
  std::uniform_real_distribution<> distrib_real;
  for (int i = 0; i < dim; i++) {
    char *vec_data = (char *)vector_data.data() + i * sizeof(float);
    float value = distrib_real(rng);
    *(float *)vec_data = value;
  }
  return vector_data;
}