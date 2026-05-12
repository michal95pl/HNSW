//
// Created by michal on 3.04.2026.
//

#ifndef HNSW_HNSW_BRUTE_FORCE_H
#define HNSW_HNSW_BRUTE_FORCE_H

#include <vector>
#include <string>
#include <HNSW.h>
#include <ground_truth_file_handler.h>
#include <aligned_utils.h>

class HNSW_benchmark {

    static constexpr uint32_t batchSize = 1000;
    std::vector<std::string> files;
    HNSW hnsw{16};

    static void mergeDistances(std::vector<GroundTruthDistance> &vec1, const std::vector<GroundTruthDistance> &vec2);
    static void getKNearestVectors(uint32_t fileId, std::vector<AlignedVector> &vectors, const AlignedVector &queryVector,
    std::vector<GroundTruthDistance> &K_Distances, uint16_t k);
public:
    explicit HNSW_benchmark(const std::vector<std::string> &files)
        : files(files) {}

    void loadVectorsToHNSW();
    void computeDistancesForQueryFile(const std::string &queryFile, uint16_t kNearestVectors) const;
    float getRecall(const std::string &groundTruthFilePath, const std::string &queryFilePath);
};



#endif
