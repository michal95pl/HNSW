#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include <vector_file_loader.h>
#include <HNSW.h>
#include <HNSW_benchmark.h>

int main() {

    HNSW_benchmark benchmark({"../data/embeddings_0.bin"});
    benchmark.loadVectorsToHNSW();
    float recall = benchmark.getRecall("../data/ground_truth.bin", "../data/query_embeddings.bin");

    std::cout << recall << std::endl;

    // benchmark.computeDistancesForQueryFile("../data/query_embeddings.bin", 3);

    return 0;
}