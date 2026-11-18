//
// Created by michal on 3.04.2026.
//

#include "hnsw/HNSW_benchmark.h"
#include "hnsw/HNSW.h"
#include "hnsw/progress_bar.h"
#include "hnsw/aligned_utils.h"
#include "hnsw/io/vector_dataset_loader.h"
#include "hnsw/io/ground_truth_dataset_loader.h"
#include <iostream>

void loadVectorsToHNSW(HNSW &hnsw, const std::string &file) {

    std::vector<AlignedVector> batch(batchSize);
    VectorDatasetLoader dataLoader(file);

    const ProgressBar progressBar(dataLoader.getNumRows(), 50, "Add vectors to hnsw");
    uint64_t vecNum = 0;

    while (dataLoader.getNextVectors(batchSize, batch)) {
        for (auto &vec : batch) {
            const uint64_t external_id = vecNum;
            hnsw.addPoint(external_id, vec.data(), 200);
            vecNum++;
        }
        progressBar.update(vecNum);
    }
}

float getRecall(HNSW &hnsw, const std::string &groundTruthFilePath, const std::string &queryFilePath, const uint8_t k) {
    GroundTruthDatasetLoader gtDataset(groundTruthFilePath);
    VectorDatasetLoader queryDataset(queryFilePath);

    std::vector<GroundTruthEntry> gtDistances = gtDataset.getAllGroundTruth();
    std::vector<AlignedVector> query_vectors = queryDataset.getAllVectors();

    if (gtDistances.empty()) {
        std::cerr << "Ground truth file is empty or could not be read." << std::endl;
        return 0;
    }

    if (query_vectors.size() != gtDistances.size()) {
        std::cerr << "Number of query vectors does not match number of ground truth entries." << std::endl;
        return 0;
    }

    uint64_t totalHits = 0;
    const uint32_t efSearch = std::max<uint32_t>(64, k);
    const size_t limit = std::min(static_cast<size_t>(k), static_cast<size_t>(gtDataset.getK()));

    for (auto &[queryId, neighborIds] : gtDistances) {
        AlignedVector& actualQuery = query_vectors[queryId];

        auto results = hnsw.search(actualQuery.data(), efSearch, limit);

        auto gtBegin = neighborIds.begin();
        auto gtEnd = gtBegin + limit;

        for (auto predictedId : results) {
            if (std::find(gtBegin, gtEnd, predictedId) != gtEnd) {
                totalHits++;
            }
        }
    }

    return static_cast<float>(totalHits) / static_cast<float>(gtDistances.size() * limit);
}