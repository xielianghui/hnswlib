#include "./common.h"

#include <netinet/in.h>

std::pair<std::unique_ptr<hnswlib::HierarchicalNSW<float>>,
          std::unique_ptr<hnswlib::L2Space>>
GenerateHnswGraph(const int dim, const int max_elements, size_t m,
                  size_t ef_construction, const int num_threads,
                  uint64_t &cost) {
  // init index
  std::unique_ptr<hnswlib::L2Space> space(new hnswlib::L2Space(dim));
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw(
      new hnswlib::HierarchicalNSW<float>(space.get(), max_elements, m,
                                          ef_construction));

  std::mt19937 rng;
  rng.seed(47);
  std::uniform_real_distribution<> distrib_real;
  float *data = new float[dim * max_elements];
  for (int i = 0; i < dim * max_elements; i++) {
    data[i] = distrib_real(rng);
  }

  // add data to index
  Timer timer;
  ParallelFor(0, max_elements, num_threads, [&](size_t i, size_t thread_id) {
    alg_hnsw->addPoint((void *)(data + i * dim), i);
  });
  cost = timer.Elapsed();

  delete[] data;
  return std::make_pair(std::move(alg_hnsw), std::move(space));
}

double GetMemoryUsageMB() {
  std::ifstream statusFile("/proc/self/status");
  std::string line;

  if (!statusFile.is_open()) {
    std::cerr << "Can't open /proc/self/status" << std::endl;
    return 0.0;
  }

  double valueMb = 0;
  while (std::getline(statusFile, line)) {
    if (line.find("VmRSS:") == 0) {
      std::string valueStr = line.substr(line.find_first_of("0123456789"));
      long valueKb = std::stol(valueStr);
      valueMb = valueKb / 1024.0;
      break;
    }
  }
  statusFile.close();
  return valueMb;
}

// codec
static void AppendFixed8(std::string &key, uint8_t i) { key.push_back(i); }

static void AppendFixed16(std::string &key, uint16_t i) {
  i = htons(i);
  key.append((char *)&i, sizeof(i));
}

static void AppendFixed32(std::string &key, uint32_t i) {
  i = htonl(i);
  key.append((char *)&i, sizeof(i));
}

static uint64_t htonll(uint64_t i) {
  if (__BYTE_ORDER == __LITTLE_ENDIAN) {
    return (((uint64_t)htonl((uint32_t)i)) << 32) |
           ((uint64_t)htonl((uint32_t)(i >> 32)));
  }
  return i;
}

static void AppendFixed64(std::string &key, uint64_t i) {
  i = htonll(i);
  key.append((char *)&i, sizeof(i));
}

static constexpr size_t kPrefixLen =
    2 * sizeof(uint64_t) + 2 * sizeof(uint16_t) + 4 * sizeof(uint8_t);
static constexpr uint8_t eKvTypeTable5 = 20;
static constexpr uint8_t kFulltextIndexTypeCode = static_cast<uint8_t>(6);

static inline void EncodePrefix(uint64_t entity_id, uint16_t table_id,
                                uint64_t uin, uint8_t field_id,
                                uint16_t sub_field_id,
                                HnswInnerKeyType key_type, std::string &key) {
  AppendFixed64(key, entity_id);
  AppendFixed8(key, eKvTypeTable5);
  AppendFixed8(key, kFulltextIndexTypeCode);
  AppendFixed16(key, table_id);
  AppendFixed64(key, uin);
  AppendFixed8(key, field_id);
  AppendFixed16(key, sub_field_id);
  AppendFixed8(key, key_type);
}

std::string EncodeHnswMetaKey(uint16_t table_id, uint64_t uin, uint8_t field_id,
                              uint16_t sub_field_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               HnswInnerKeyType::kMeta, key);
  return key;
}

std::string EncodeHnswLinkListKey(uint16_t table_id, uint64_t uin,
                                  uint8_t field_id, uint16_t sub_field_id,
                                  int internal_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               HnswInnerKeyType::kLinkList, key);
  AppendFixed32(key, internal_id);
  return key;
}

std::string EncodeHnswLinkListBeginKey(uint16_t table_id, uint64_t uin,
                                       uint8_t field_id,
                                       uint16_t sub_field_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               HnswInnerKeyType::kLinkList, key);
  return key;
}

std::string EncodeHnswLinkListEndKey(uint16_t table_id, uint64_t uin,
                                     uint8_t field_id, uint16_t sub_field_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               (HnswInnerKeyType)(HnswInnerKeyType::kLinkList + 1), key);
  return key;
}

std::string EncodeHnswLevel0DataKey(uint16_t table_id, uint64_t uin,
                                    uint8_t field_id, uint16_t sub_field_id,
                                    int internal_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               HnswInnerKeyType::kLevel0Data, key);
  AppendFixed32(key, internal_id);
  return key;
}

std::string EncodeHnswLevel0DataBeginKey(uint16_t table_id, uint64_t uin,
                                         uint8_t field_id,
                                         uint16_t sub_field_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               HnswInnerKeyType::kLevel0Data, key);
  return key;
}

std::string EncodeHnswLevel0DataEndKey(uint16_t table_id, uint64_t uin,
                                       uint8_t field_id,
                                       uint16_t sub_field_id) {
  std::string key;
  EncodePrefix(0, table_id, uin, field_id, sub_field_id,
               (HnswInnerKeyType)(HnswInnerKeyType::kLevel0Data + 1), key);
  return key;
}