#include <iostream>
#include <random>
#include "hnsw/HNSW_benchmark.h"
#include "hnsw/HNSW.h"
#include "hnsw/io/ground_truth_dataset_loader.h"

static HNSW hnsw;

int main() {

    loadVectorsToHNSW(hnsw, "../data/pca256_base.bin");
    const float recall = getRecall(hnsw, "../data/ground_truth.bin", "../data/pca256_query.bin", 10);
    std::cout << recall << std::endl;

    return 0;
}