//
// Created by michal on 2.04.2026.
//

#include <vector_file_loader.h>
#include <fstream>
#include <iostream>

#include "aligned_utils.h"

VectorFileLoader::VectorFileLoader(const std::string& path) : file(path, std::ios::binary) {
    if (!(file.is_open() && loadHeader())) {
        std::cout << "Failed to open file" << std::endl;
    }
}

VectorFileLoader::~VectorFileLoader() {
    if (file.is_open()) {
        file.close();
    }
}

bool VectorFileLoader::loadHeader() {
    if (!file.is_open()) return false;

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(&numVectors), 4);
    file.read(reinterpret_cast<char*>(&dimension), 4);

    return file.good();
}

bool VectorFileLoader::readNextVector(AlignedVector& outVec) {
    if (!file.is_open() || file.eof()) return false;

    outVec.resize(dimension);
    std::streamsize bytesToRead = dimension * sizeof(float);
    file.read(reinterpret_cast<char*>(outVec.data()), bytesToRead);

    if (file.gcount() != bytesToRead) {
        return false;
    }

    return true;
}

bool VectorFileLoader::readNextVectorBatch(std::vector<AlignedVector>& outBatch, const uint32_t batchSize) {
    if (!file.is_open() || file.eof()) return false;

    std::vector<float> tempFlatBuff(batchSize * dimension);
    const std::streamsize bytesToRead = batchSize * dimension * sizeof(float);

    file.read(reinterpret_cast<char*>(tempFlatBuff.data()), bytesToRead);

    const std::streamsize bytesRead = file.gcount();
    const size_t actualVectors = bytesRead / (dimension * sizeof(float));

    if (actualVectors == 0) return false;

    outBatch.clear();
    outBatch.reserve(actualVectors);

    for (size_t i = 0; i < actualVectors; ++i) {
        auto start = tempFlatBuff.begin() + (i * dimension);
        auto end = start + dimension;
        outBatch.emplace_back(start, end);
    }

    return true;
}

bool VectorFileLoader::seekToVector(uint32_t index) {
    if (!file.is_open() || index >= numVectors) return false;

    const std::streamoff offset = 8 + (static_cast<std::streamoff>(index) * dimension * sizeof(float));
    file.clear();

    file.seekg(offset, std::ios::beg);

    return file.good();
}
