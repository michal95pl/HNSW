//
// Created by michal on 10.11.2026.
//

#include "hnsw/io/vector_dataset_loader.h"
#include <cstring>
#include <iostream>
#include "hnsw/aligned_utils.h"

VectorDatasetLoader::VectorDatasetLoader(const std::string& path)
    : DatasetFileLoader(path, MAGIC, HEADER_SIZE_BYTES) {
    uint32_t typeValue = 0;
    if (!readHeaderValue(dimension) || !readHeaderValue(typeValue)) return;

    if (dimension == 0) {
        std::cerr << "Vector dimension is 0" << std::endl;
        return;
    }

    switch (static_cast<VectorType>(typeValue)) {
        case VectorType::RAW_FLOAT:
        case VectorType::COMPRESSED_FLOAT:
            vectorType = static_cast<VectorType>(typeValue);
            setRowSize(static_cast<uint64_t>(dimension) * sizeof(float));
            break;
        case VectorType::BINARY_HAMMING:
            vectorType = VectorType::BINARY_HAMMING;
            setRowSize((static_cast<uint64_t>(dimension) + 7) / 8);
            break;
        default:
            std::cerr << "Unknown vector type: " << std::to_string(typeValue) << std::endl;
    }
}

std::vector<AlignedVector> VectorDatasetLoader::getAllVectors() {
    return readAllDecoded<AlignedVector>([this](const char* row, AlignedVector& out) { decodeRow(row, out); });
}

bool VectorDatasetLoader::getNextVectors(const uint64_t n, std::vector<AlignedVector>& out) {
    return readNextDecoded(n, out, [this](const char* row, AlignedVector& vec) { decodeRow(row, vec); });
}

void VectorDatasetLoader::decodeRow(const char* row, AlignedVector& out) const {
    out.resize(dimension);

    if (vectorType == VectorType::BINARY_HAMMING) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(row);
        for (uint32_t i = 0; i < dimension; ++i) {
            out[i] = static_cast<float>((bytes[i >> 3] >> (7 - (i & 7))) & 1u);
        }
    } else {
        std::memcpy(out.data(), row, static_cast<size_t>(dimension) * sizeof(float));
    }
}
