//
// Created by michal on 3.04.2026.
//

#include <HNSW_benchmark.h>
#include <HNSW.h>
#include <vector_file_loader.h>
#include <progress_bar.h>
#include <ground_truth_file_handler.h>
#include <queue>
#include <iostream>

void HNSW_benchmark::loadVectorsToHNSW() {
    uint64_t fileNum = 0;
    std::vector<AlignedVector> batch(batchSize);

    for (auto &file : files) {
        VectorFileLoader loader(file);
        ProgressBar progressBar(loader.getCount(), 50, "Add vectors to database");
        uint64_t vecNum = 0;
        while (loader.readNextVectorBatch(batch, batchSize)) {
            for (auto &vec : batch) {
                const uint64_t external_id = (fileNum << 32) | (vecNum & 0xFFFFFFFF);
                hnsw.addPoint(external_id, std::move(vec), 30, 60);
                vecNum++;
            }
            progressBar.update(vecNum);
        }
        fileNum++;
    }
}

float HNSW_benchmark::getRecall(const std::string &groundTruthFilePath, const std::string &queryFilePath) {
    GroundTruthFileHandler gtHandler(groundTruthFilePath);
    std::vector<GroundTruthDistance> gtDistances = gtHandler.getAllVectors();

    if (gtDistances.empty()) {
        std::cout << "Ground truth file is empty or could not be read." << std::endl;
        return 0;
    }
    VectorFileLoader queryLoader(queryFilePath);
    std::vector<AlignedVector> query_vectors(queryLoader.getCount());
    queryLoader.readNextVectorBatch(query_vectors, queryLoader.getCount());

    if (query_vectors.size() != gtDistances.size() / gtHandler.getK()) {
        std::cout << "Number of query vectors does not match number of ground truth entries." << std::endl;
        return 0;
    }

    uint32_t counter = 0;
    for (uint32_t query_id = 0; query_id < query_vectors.size(); ++query_id) {
        uint64_t hnswResult = hnsw.search(std::move(query_vectors[query_id]), 60, 1)[0];
        uint64_t truthResult = gtDistances[query_id * gtHandler.getK()].id;
        if (hnswResult == truthResult) {
            counter++;
        }
    }

    if (counter == 0) {
        return 0;
    }

    return static_cast<float>(counter) / static_cast<float>(query_vectors.size());
}

void HNSW_benchmark::getKNearestVectors(const uint32_t fileId, std::vector<AlignedVector> &vectors, const AlignedVector &queryVector,
    std::vector<GroundTruthDistance> &K_Distances, const uint16_t k) {

    std::priority_queue<GroundTruthDistance> maxHeap;
    K_Distances.clear();
    uint32_t vectorId = 0;

    for (auto &vector : vectors) {
        const float distance = HNSW::calculateDistance(vector, queryVector);
        const uint64_t external_id = (static_cast<uint64_t>(fileId) << 32) | (vectorId & 0xFFFFFFFF);
        if (maxHeap.size() < k) {
            maxHeap.push(GroundTruthDistance(external_id, distance));
        } else if (distance < maxHeap.top().distance) {
            maxHeap.pop();
            maxHeap.push(GroundTruthDistance(external_id, distance));
        }
        vectorId++;
    }

    while (!maxHeap.empty()) {
        K_Distances.push_back(maxHeap.top());
        maxHeap.pop();
    }
    std::reverse(K_Distances.begin(), K_Distances.end());
}

void HNSW_benchmark::mergeDistances(std::vector<GroundTruthDistance> &vec1, const std::vector<GroundTruthDistance> &vec2) {
    std::vector<GroundTruthDistance> merged;
    const size_t k = vec1.size();
    uint16_t i = 0, j = 0;

    while (merged.size() < k && (i < vec1.size() || j < vec2.size())) {
        if (i < vec1.size() && (j >= vec2.size() || vec1[i].distance <= vec2[j].distance)) {
            merged.push_back(vec1[i++]);
        } else {
            merged.push_back(vec2[j++]);
        }
    }
    vec1 = std::move(merged);
}

void HNSW_benchmark::computeDistancesForQueryFile(const std::string &queryFile, uint16_t kNearestVectors) const {
    VectorFileLoader queryLoader(queryFile);
    std::vector<AlignedVector> query_vectors(queryLoader.getCount());
    queryLoader.readNextVectorBatch(query_vectors, queryLoader.getCount());

    for (auto &vector : query_vectors) {
        HNSW::normalizeVector(vector);
    }

    GroundTruthFileHandler gtHandler("../data/ground_truth.bin");
    gtHandler.createFile(kNearestVectors);

    std::vector<std::vector<GroundTruthDistance>> query_k_distances(query_vectors.size());

    std::vector<AlignedVector> file_vectors;
    uint32_t file_id = 0;
    for (auto &file : files) {
        VectorFileLoader loader(file);
        file_vectors.resize(loader.getCount());
        loader.readNextVectorBatch(file_vectors, loader.getCount());

        ProgressBar progressBar(query_vectors.size(), 50, "Computing distances for dataset: " + file);

        #pragma omp parallel for schedule(dynamic)
        for (auto &vector : file_vectors) {
            HNSW::normalizeVector(vector);
        }

        #pragma omp parallel for schedule(dynamic)
        for (uint32_t query_id = 0; query_id < query_vectors.size(); ++query_id) {
            std::vector<GroundTruthDistance> temp_distances;
            getKNearestVectors(file_id, file_vectors, query_vectors[query_id], temp_distances, kNearestVectors);

            if (query_k_distances[query_id].empty()) {
                query_k_distances[query_id] = std::move(temp_distances);
            } else {
                mergeDistances(query_k_distances[query_id], temp_distances);
            }

            if (query_id % 20 == 0 || query_id == query_vectors.size() - 1) {
                #pragma omp critical(progress_bar_update)
                {
                    progressBar.update(query_id + 1);
                }
            }
        }
        file_id++;
    }

    for (const auto& query_distance : query_k_distances) {
        gtHandler.appendQueryVectorDistances(query_distance);
    }
}