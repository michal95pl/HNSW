//
// Created by michal on 2.04.2026.
//

#ifndef HNSW_VECTOR_FILE_LOADER_H
#define HNSW_VECTOR_FILE_LOADER_H

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <aligned_utils.h>

class VectorFileLoader {
    uint32_t numVectors{};
    uint32_t dimension{};
    std::ifstream file;

    bool loadHeader();
public:
    explicit VectorFileLoader(const std::string& filePath);
    ~VectorFileLoader();
    
    bool readNextVector(AlignedVector& vector);
    bool readNextVectorBatch(std::vector<AlignedVector>& batch, uint32_t batchSize);
    bool seekToVector(uint32_t index);

    uint32_t getCount() const { return numVectors; }
    uint32_t getDim() const { return dimension; }
};

#endif
