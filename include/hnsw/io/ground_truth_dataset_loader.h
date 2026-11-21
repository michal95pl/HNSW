//
// Created by michal on 14.11.2026.
//

#ifndef HNSW_GROUND_TRUTH_DATASET_LOADER_H
#define HNSW_GROUND_TRUTH_DATASET_LOADER_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "dataset_file_loader.h"

struct GroundTruthEntry {
    int32_t queryId = -1;
    std::vector<int32_t> neighborIds;
};

class GroundTruthDatasetLoader : public DatasetFileLoader {
public:
    explicit GroundTruthDatasetLoader(const std::string& path);

    [[nodiscard]] uint32_t getK() const { return k; }

    [[nodiscard]] uint64_t getNumQueries() const { return getNumRows(); }

    [[nodiscard]] std::vector<GroundTruthEntry> getAllGroundTruth();


    [[nodiscard]] std::vector<GroundTruthEntry> getNextGroundTruth(uint64_t n);

    bool getNextGroundTruth(uint64_t n, std::vector<GroundTruthEntry>& out);

private:
    static constexpr std::string_view MAGIC = "GTRF";
    static constexpr uint32_t HEADER_SIZE_BYTES = 8; // 'GTRF' (4B) + k (4B)

    uint32_t k = 0;

    void decodeRow(const char* row, GroundTruthEntry& out) const;
};

#endif
