//
// Created by michal on 10.11.2026.
//

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <hnsw/aligned_utils.h>
#include "dataset_file_loader.h"


enum class VectorType : uint32_t {
    RAW_FLOAT = 1,
    COMPRESSED_FLOAT = 2,
    BINARY_HAMMING = 3
};

class VectorDatasetLoader : public DatasetFileLoader {
public:
    explicit VectorDatasetLoader(const std::string& path);

    [[nodiscard]] uint32_t getVectorSize() const { return dimension; }

    [[nodiscard]] uint64_t getLengthVector() const { return getNumRows(); }

    [[nodiscard]] VectorType getVectorType() const { return vectorType; }

    [[nodiscard]] std::vector<AlignedVector> getAllVectors();

    bool getNextVectors(uint64_t n, std::vector<AlignedVector>& out);

private:
    static constexpr std::string_view MAGIC = "VECF";
    static constexpr uint32_t HEADER_SIZE_BYTES = 12; // 'VECF' (4B) + dim (4B) + type (4B)

    uint32_t dimension = 0;
    VectorType vectorType = VectorType::RAW_FLOAT;

    void decodeRow(const char* row, AlignedVector& out) const;
};
