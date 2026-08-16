//
// Created by michal on 3.04.2026.
//

#include <algorithm>
#include <cstring>
#include <HNSW.h>
#include <random>
#include <queue>
#include <unordered_set>
#include <immintrin.h>
#include <span>
#include <array>
#include <iostream>

HNSW::HNSW(const uint8_t M) :
externalIdsPtr(std::make_unique<uint64_t[]>(HNSW_SIZE)),
upper_neighbors_countsPtr(std::make_unique<uint8_t[]>(HNSW_SIZE * (HNSW_MAX_LEVEL-1))),
upper_neighborsPtr(std::make_unique<uint32_t[]>((HNSW_MAX_LEVEL-1) * HNSW_SIZE * (HNSW_MAX_NEIGHBORS+40))),
layer0_countsPtr(std::make_unique<uint8_t[]>(HNSW_SIZE)),
layer0_neighborsPtr(std::make_unique<uint32_t[]>(HNSW_SIZE * (HNSW_MAX_NEIGHBORS+40))),
embeddingsPtr(make_aligned_array<float>(HNSW_SIZE * EMB_DIM)),
layer0_neighborsView(layer0_neighborsPtr.get()),
upper_neighbors_countsView(upper_neighbors_countsPtr.get()),
upper_layer_neighborsView(upper_neighborsPtr.get()),
embeddingsView(embeddingsPtr.get()),
versionedSetArray(HNSW_SIZE, 0)
{
    ml = 1.0 / std::log(static_cast<double>(M));
}

uint8_t HNSW::getRandomLevel() const {
    // unsafe for threads
    static std::mt19937 gen(42);
    static std::uniform_real_distribution dis(0.0, 1.0);

    double r = dis(gen);
    if (r == 0) r = 1e-10;

    auto level = static_cast<uint8_t>(-std::log(r) * ml);
    if (level > HNSW_MAX_LEVEL) level = HNSW_MAX_LEVEL;
    return level;
}

__attribute__((target("avx512f,fma")))
void HNSW::normalizeVectorAVX512(float* vec) {

    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();

    for (size_t i = 0; i < EMB_DIM; i += 64) {
        const __m512 v1 = _mm512_load_ps(&vec[i]);
        const __m512 v2 = _mm512_load_ps(&vec[i + 16]);
        const __m512 v3 = _mm512_load_ps(&vec[i + 32]);
        const __m512 v4 = _mm512_load_ps(&vec[i + 48]);

        sum1 = _mm512_fmadd_ps(v1, v1, sum1);
        sum2 = _mm512_fmadd_ps(v2, v2, sum2);
        sum3 = _mm512_fmadd_ps(v3, v3, sum3);
        sum4 = _mm512_fmadd_ps(v4, v4, sum4);
    }

    const __m512 total_sum_v = _mm512_add_ps(_mm512_add_ps(sum1, sum2), _mm512_add_ps(sum3, sum4));
    const float sum_sq = _mm512_reduce_add_ps(total_sum_v);

    if (sum_sq < 1e-12f) return;

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    const __m512 inv_norm_v = _mm512_set1_ps(inv_norm);

    for (size_t i = 0; i < EMB_DIM; i += 64) {
        __m512 v1 = _mm512_load_ps(&vec[i]);
        __m512 v2 = _mm512_load_ps(&vec[i + 16]);
        __m512 v3 = _mm512_load_ps(&vec[i + 32]);
        __m512 v4 = _mm512_load_ps(&vec[i + 48]);

        _mm512_store_ps(&vec[i],      _mm512_mul_ps(v1, inv_norm_v));
        _mm512_store_ps(&vec[i + 16], _mm512_mul_ps(v2, inv_norm_v));
        _mm512_store_ps(&vec[i + 32], _mm512_mul_ps(v3, inv_norm_v));
        _mm512_store_ps(&vec[i + 48], _mm512_mul_ps(v4, inv_norm_v));
    }
}

void HNSW::addPoint(const uint64_t externalID, const float* embedding, const uint16_t ef) {
    const uint32_t newNodeID = embeddingsSize++;
    const uint8_t level = getRandomLevel();

    std::memcpy(&embeddingsView[newNodeID, 0], embedding, EMB_DIM * sizeof(float));
    normalizeVectorAVX512(&embeddingsView[newNodeID, 0]);

    externalIdsPtr[newNodeID] = externalID;

    // initialization
    if (embeddingsSize == 1) {
        entryPointID = 0;
        actualMaxLevel = level;
        return;
    }

    uint32_t currNodeID = entryPointID;
    uint32_t foundedNeighbors[ef];

    // search best entry point for each layer above the new node's level
    for (int l = actualMaxLevel; l > level; --l) {
        if (searchLayer<false>(&embeddingsView[newNodeID, 0], currNodeID, 1, l, foundedNeighbors)) {
            currNodeID = foundedNeighbors[0];
        }
    }

    for (int l = std::min(level, actualMaxLevel); l >= 1; --l) {
        const uint16_t num_neighbors = searchLayer<false>(&embeddingsView[newNodeID, 0], currNodeID, ef, l, foundedNeighbors);
        if (!num_neighbors) continue;

        for (uint16_t i = 0; i < num_neighbors; ++i) {
            const uint32_t neighborID = foundedNeighbors[i];

            const uint8_t countIdx = upper_neighbors_countsView[neighborID, l - 1]++;
            upper_layer_neighborsView[neighborID, l - 1, countIdx] = newNodeID;

            const uint8_t newCountIdx = upper_neighbors_countsView[newNodeID, l - 1]++;
            upper_layer_neighborsView[newNodeID, l - 1, newCountIdx] = neighborID;

            shrinkNeighbors<false>(neighborID, l, HNSW_MAX_NEIGHBORS);
        }
        currNodeID = foundedNeighbors[0];
        shrinkNeighbors<false>(newNodeID, l, HNSW_MAX_NEIGHBORS);
    }

    const uint16_t num_neighbors = searchLayer<true>(&embeddingsView[newNodeID, 0], currNodeID, ef, 0, foundedNeighbors);

    if (num_neighbors > 0) {
        for (uint16_t i = 0; i < num_neighbors; ++i) {
            const uint32_t neighborID = foundedNeighbors[i];

            const uint8_t countIdx = layer0_countsPtr[neighborID]++;
            layer0_neighborsView[neighborID, countIdx] = newNodeID;

            const uint8_t newCountIdx = layer0_countsPtr[newNodeID]++;
            layer0_neighborsView[newNodeID, newCountIdx] = neighborID;

            shrinkNeighbors<true>(neighborID, 0, HNSW_MAX_NEIGHBORS);
        }
        shrinkNeighbors<true>(newNodeID, 0, HNSW_MAX_NEIGHBORS);
    }

    if (level > actualMaxLevel) {
        entryPointID = newNodeID;
        actualMaxLevel = level;
    }
}

__attribute__((target("avx512f,fma")))
float HNSW::calculateDistanceAVX512(const float* vec1, const float* vec2) {
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();

    for (size_t i = 0; i < EMB_DIM; i += 64) {
        sum1 = _mm512_fmadd_ps(_mm512_load_ps(&vec1[i]),      _mm512_load_ps(&vec2[i]),      sum1);
        sum2 = _mm512_fmadd_ps(_mm512_load_ps(&vec1[i + 16]), _mm512_load_ps(&vec2[i + 16]), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_load_ps(&vec1[i + 32]), _mm512_load_ps(&vec2[i + 32]), sum3);
        sum4 = _mm512_fmadd_ps(_mm512_load_ps(&vec1[i + 48]), _mm512_load_ps(&vec2[i + 48]), sum4);
    }

    const __m512 total_sum = _mm512_add_ps(_mm512_add_ps(sum1, sum2), _mm512_add_ps(sum3, sum4));

    const float dot = _mm512_reduce_add_ps(total_sum);
    const float dist = 1.0f - dot;
    return dist < 0 ? 0 : dist;
}

std::vector<uint64_t> HNSW::search(float* query_emb, const uint16_t ef, const uint16_t k) {
    uint32_t actualEntryPoint = entryPointID;

    normalizeVectorAVX512(query_emb);

    uint32_t founded_neighbors[ef];
    for (int i = actualMaxLevel; i >= 1; --i) {
        if (searchLayer<false>(query_emb, actualEntryPoint, 1, i, founded_neighbors)) {
            actualEntryPoint = founded_neighbors[0];
        }
    }

    const uint16_t numNeighbours = searchLayer<true>(query_emb, actualEntryPoint, ef, 0, founded_neighbors);
    std::vector<uint64_t> result;
    const uint16_t actualK = std::min(k, numNeighbours);
    result.reserve(actualK);
    for (uint16_t i = 0; i < actualK; i++) {
        result.push_back(externalIdsPtr[founded_neighbors[i]]);
    }
    return result;
}