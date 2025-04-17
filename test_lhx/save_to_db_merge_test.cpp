#include "common.h"

#include <netinet/in.h>

#include "leveldb/db.h"
#include "leveldb/env.h"
#include "leveldb/options.h"
#include "leveldb/write_batch.h"

static const int KM = 16;
static const int KEFConstruction = 100;
static const uint16_t kTableId = 666;
static const uint64_t kUin = 888888;
static const uint8_t kFieldId = 6;
static const uint16_t kSubFieldId = 0;
static const std::string KDbname = "./index_db";

static size_t kLoadThreadCnt = 10;
static size_t kMergeKeySize = 100;

static void
Save2Db(leveldb::DB *db,
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> &alg_hnsw) {
  Timer timer;

  leveldb::WriteBatch wb;
  std::string meta_key =
      EncodeHnswMetaKey(kTableId, kUin, kFieldId, kSubFieldId);
  auto meta_value = GetHnswMetaValue(alg_hnsw);
  auto s = db->Put(leveldb::WriteOptions(), meta_key, meta_value.Serialize());
  if (!s.ok()) {
    std::cout << "Failed to write db, " << s.ToString() << std::endl;
  }

  char *link_list_data = nullptr;
  size_t link_list_data_size = 0;
  for (size_t i = 0; i < alg_hnsw->max_elements_; ++i) {
    GetHnswLinkListData(alg_hnsw, i, link_list_data, link_list_data_size);
    if (link_list_data_size > 0) {
      leveldb::Slice link_list_value(link_list_data, link_list_data_size);
      wb.Put(EncodeHnswLinkListKey(kTableId, kUin, kFieldId, kSubFieldId, i),
             link_list_value);
    }
  }

  size_t count = 0;
  for (size_t i = 0; i < alg_hnsw->max_elements_;) {
    size_t begin = i;
    size_t end = begin + kMergeKeySize;
    std::string key =
        EncodeHnswLevel0DataKey(kTableId, kUin, kFieldId, kSubFieldId, begin);
    if (end > alg_hnsw->max_elements_) {
      end = alg_hnsw->max_elements_;
    }
    i = end;

    char *data = nullptr;
    size_t data_size = 0;
    GetHnswLevel0Data(alg_hnsw, begin, data, data_size);
    leveldb::Slice level0_data(data, (end - begin) *
                                         alg_hnsw->size_data_per_element_);
    wb.Put(key, level0_data);
    ++count;
  }

  s = db->Write(leveldb::WriteOptions(), &wb);
  if (!s.ok()) {
    std::cout << "Failed to write db, " << s.ToString() << std::endl;
  }
  std::cout << "DB SAVE: total_cost=" << timer.Elapsed()
            << " ms, data_level0_key_size=" << count << std::endl;
}

static inline int GetInternalIdFromKey(leveldb::Slice &key) {
  static constexpr size_t kPrefixLen =
      2 * sizeof(uint64_t) + 2 * sizeof(uint16_t) + 4 * sizeof(uint8_t);
  return ntohl(*(uint32_t *)(key.data() + kPrefixLen));
}

static inline HnswInnerKeyType GetTypeFromKey(leveldb::Slice &key) {
  static constexpr size_t kPrefixLen =
      2 * sizeof(uint64_t) + 2 * sizeof(uint16_t) + 3 * sizeof(uint8_t);

  return *(HnswInnerKeyType *)(key.data() + kPrefixLen);
}

static std::pair<std::unique_ptr<hnswlib::HierarchicalNSW<float>>,
                 std::unique_ptr<hnswlib::L2Space>>
LoadFromDb(leveldb::DB *db, uint16_t table_id, uint64_t uin, uint8_t field_id,
           uint16_t sub_field_id) {
  Timer timer;
  std::string meta_key =
      EncodeHnswMetaKey(kTableId, kUin, kFieldId, kSubFieldId);
  std::string meta_value_str;
  auto s = db->Get(leveldb::ReadOptions(), meta_key, &meta_value_str);
  if (!s.ok()) {
    std::cout << "Failed to get meta value, " << s.ToString() << std::endl;
    return std::make_pair(nullptr, nullptr);
  }

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

  std::vector<std::thread> threads;

  // malloc locks
  threads.emplace_back(std::thread([&alg_hnsw]() {
    std::vector<std::mutex>(alg_hnsw->MAX_LABEL_OPERATION_LOCKS)
        .swap(alg_hnsw->label_op_locks_);
    std::vector<std::mutex>(alg_hnsw->max_elements_)
        .swap(alg_hnsw->link_list_locks_);
    alg_hnsw->visited_list_pool_.reset(
        new hnswlib::VisitedListPool(1, alg_hnsw->max_elements_));
  }));

  // load data level0
  alg_hnsw->data_level0_memory_ = (char *)malloc(
      alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
  if (alg_hnsw->data_level0_memory_ == nullptr) {
    throw std::runtime_error(
        "Not enough memory: loadIndex failed to allocate level0");
  }

  // load link list
  alg_hnsw->linkLists_ =
      (char **)malloc(sizeof(void *) * alg_hnsw->max_elements_);
  if (alg_hnsw->linkLists_ == nullptr) {
    throw std::runtime_error(
        "Not enough memory: loadIndex failed to allocate linklists");
  }

  std::atomic<uint64_t> count{0};
  uint64_t gap = alg_hnsw->max_elements_ / kLoadThreadCnt;
  uint64_t start = 0;
  for (size_t i = 0; i < kLoadThreadCnt; ++i) {
    uint64_t begin_id = start;
    std::string data_level0_begin_key =
        EncodeHnswLevel0DataKey(table_id, uin, field_id, sub_field_id, start);
    std::string link_list_begin_key =
        EncodeHnswLinkListKey(table_id, uin, field_id, sub_field_id, start);
    if (i == kLoadThreadCnt - 1) {
      start = alg_hnsw->max_elements_;
    } else {
      start += gap;
    }
    std::string data_level0_end_key =
        EncodeHnswLevel0DataKey(table_id, uin, field_id, sub_field_id, start);
    std::string link_list_end_key =
        EncodeHnswLinkListKey(table_id, uin, field_id, sub_field_id, start);
    uint64_t end_id = start;

    threads.emplace_back(std::thread(
        [&db, &alg_hnsw, &count, begin_id, end_id, data_level0_begin_key,
         data_level0_end_key, link_list_begin_key, link_list_end_key, table_id,
         uin, field_id, sub_field_id] {
          auto reado_options = leveldb::ReadOptions();
          reado_options.fill_cache = false;
          auto *iter = db->NewIterator(reado_options);
          std::unique_ptr<leveldb::Iterator> iter_deleter(iter);

          int id = 0;
          for (iter->Seek(link_list_begin_key); iter->Valid(); iter->Next()) {
            auto key = iter->key();
            auto value = iter->value();
            if (GetTypeFromKey(key) != HnswInnerKeyType::kLinkList) {
              break;
            }
            id = GetInternalIdFromKey(key);
            if (id >= end_id) {
              break;
            }
            alg_hnsw->element_levels_[id] =
                value.size() / alg_hnsw->size_links_per_element_;
            alg_hnsw->linkLists_[id] = (char *)malloc(value.size());
            memcpy(alg_hnsw->linkLists_[id], value.data(), value.size());
          }

          id = 0;
          for (iter->Seek(data_level0_begin_key); iter->Valid(); iter->Next()) {
            auto key = iter->key();
            auto value = iter->value();
            if (GetTypeFromKey(key) != HnswInnerKeyType::kLevel0Data) {
              break;
            }
            id = GetInternalIdFromKey(key);
            if (id >= end_id) {
              break;
            }
            count.fetch_add(kMergeKeySize, std::memory_order_relaxed);
            memcpy(alg_hnsw->get_linklist0(id), value.data(), value.size());
          }
        }));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // only used for update
  //   for (size_t i = 0; i < alg_hnsw->max_elements_; ++i) {
  //     alg_hnsw->label_lookup_[alg_hnsw->getExternalLabel(i)] = i;
  //   }

  std::cout << "DB LOAD: thread_count=" << kLoadThreadCnt
            << ", total_cost=" << timer.Elapsed() << " ms, count=" << count
            << ", merge_key_size=" << kMergeKeySize << std::endl;

  assert(count == alg_hnsw->cur_element_count);
  return std::make_pair(std::move(alg_hnsw), std::move(space));
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " [index_file_path] [load_thread_cnt] [merge_key_size]"
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

  if (argc > 2) {
    kLoadThreadCnt = atoi(argv[2]);
  }
  if (argc > 3) {
    kMergeKeySize = atoi(argv[3]);
  }

  Timer timer;
  hnswlib::L2Space space(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw(
      new hnswlib::HierarchicalNSW<float>(&space, filename));
  std::cout << "FILE LOAD: total_cost=" << timer.ElapsedMicroseconds()
            << " μs, mem_usage=" << GetMemoryUsageMB() << " MB" << std::endl;

  // open db
  leveldb::DestroyDB(KDbname, leveldb::Options());
  leveldb::DB *db = nullptr;
  leveldb::Options opts;
  opts.create_if_missing = true;
  auto s = leveldb::DB::Open(opts, KDbname, &db);
  if (!s.ok()) {
    std::cout << "Failed to open db, " << s.ToString() << std::endl;
  }
  std::unique_ptr<leveldb::DB> db_deleter(db);

  // save to db
  Save2Db(db, alg_hnsw);

  // load from db
  auto res = LoadFromDb(db, kTableId, kUin, kFieldId, kSubFieldId);
  auto &reload_alg_hnsw = res.first;

  // check valid
  assert(GetHnswMetaValue(alg_hnsw) == GetHnswMetaValue(reload_alg_hnsw));
  std::string query_data = GetRandomVectorData(dim);
  auto top100 = alg_hnsw->searchKnnCloserFirst((void *)query_data.data(), 100);
  auto reload_top100 =
      reload_alg_hnsw->searchKnnCloserFirst((void *)query_data.data(), 100);
  assert(top100 == reload_top100);

  return 0;
}