#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <thread>

#include "../hnswlib/hnswlib.h"

std::unique_ptr<hnswlib::HierarchicalNSW<float>>
GenerateHnswGraph(const int dim, const int max_elements, const int num_threads,
                  uint64_t &cost);

class Timer {
public:
  Timer() : startTimePoint(std::chrono::high_resolution_clock::now()) {}

  void reset() { startTimePoint = std::chrono::high_resolution_clock::now(); }

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