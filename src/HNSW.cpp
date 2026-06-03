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
embeddingsPtr(make_aligned_array<float>(HNSW_SIZE * EMB_DIM)),
nodeMetadataPtr(std::make_unique<NodeMetadata[]>(HNSW_SIZE)),
embeddingsView(embeddingsPtr.get(), HNSW_SIZE),
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

void HNSW::shrinkNeighbors(const uint32_t neighbor_id, const uint8_t layer, const uint16_t maxNeighbors) {

    if (nodeMetadataPtr[neighbor_id].neighborCounts[layer] <= maxNeighbors) {
        return;
    }

    const uint8_t neighborCount = nodeMetadataPtr[neighbor_id].neighborCounts[layer];

    std::pair<float, uint32_t> dist_pairs[neighborCount];

    for (uint16_t i=0; i < neighborCount; ++i) {
        const uint32_t neighborId = nodeMetadataPtr[neighbor_id].neighbors[layer][i];
        dist_pairs[i].second = neighborId;
        dist_pairs[i].first = calculateDistanceAVX512(&embeddingsView[neighbor_id, 0], &embeddingsView[neighborId, 0]);
    }

    auto data_view = std::span(dist_pairs, neighborCount);
    std::ranges::nth_element(data_view, data_view.begin() + maxNeighbors, std::less{}, &std::pair<float, uint32_t>::first);

    nodeMetadataPtr[neighbor_id].neighborCounts[layer] = maxNeighbors;

    for (uint16_t i = 0; i < maxNeighbors; ++i) {
        nodeMetadataPtr[neighbor_id].neighbors[layer][i] = dist_pairs[i].second;
    }
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

    nodeMetadataPtr[newNodeID].externalId = externalID;
    nodeMetadataPtr[newNodeID].layer = level;

    // initialization
    if (embeddingsSize == 1) {
        entryPointID = 0;
        maxLevel = level;
        return;
    }

    uint32_t currNodeID = entryPointID;
    uint32_t foundedNeighbors[ef];

    // search best entry point for each layer above the new node's level
    for (int l = maxLevel; l > level; --l) {
        if (searchLayer(&embeddingsView[newNodeID, 0], currNodeID, 1, l, foundedNeighbors)) {
            currNodeID = foundedNeighbors[0];
        }
    }

    for (int l = std::min(level, maxLevel); l >= 0; --l) {
        const uint16_t num_neighbors = searchLayer(&embeddingsView[newNodeID, 0], currNodeID, ef, l, foundedNeighbors);
        if (!num_neighbors) continue;

        for (uint16_t i = 0; i < num_neighbors; ++i) {
            const uint32_t neighborID = foundedNeighbors[i];
            nodeMetadataPtr[newNodeID].neighbors[l][nodeMetadataPtr[newNodeID].neighborCounts[l]++] = neighborID;
            nodeMetadataPtr[neighborID].neighbors[l][nodeMetadataPtr[neighborID].neighborCounts[l]++] = newNodeID;
            shrinkNeighbors(neighborID, l, HNSW_MAX_NEIGHBORS);
        }
        currNodeID = foundedNeighbors[0];
        shrinkNeighbors(newNodeID, l, HNSW_MAX_NEIGHBORS);
    }

    if (level > maxLevel) {
        entryPointID = newNodeID;
        maxLevel = level;
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

uint16_t HNSW::searchLayer(const float* query_emb, uint32_t entry_point, const uint16_t ef, const uint8_t level, uint32_t* founded_neighbors) {

    std::array<BufferElement, 201> buffer{};
    uint16_t buffer_size = 0;

    const float dist = calculateDistanceAVX512(query_emb, &embeddingsView[entry_point, 0]);

    buffer[buffer_size++] = {entry_point, dist, false};
    versionedSetArray[entry_point] = actualSetVersion;

    uint32_t curr_idx = 0;
    while (curr_idx < buffer_size) {

        if (curr_idx + 1 < buffer_size) {
            uint32_t nextNodeID = buffer[curr_idx + 1].id;
            _mm_prefetch(&nodeMetadataPtr[nextNodeID], _MM_HINT_T0);
        }

        if (buffer_size >= ef && buffer[curr_idx].dist > buffer[buffer_size - 1].dist) break;

        if (!buffer[curr_idx].expanded) {
            buffer[curr_idx].expanded = true;
            const u_int32_t currNodeID = buffer[curr_idx].id;

            const uint8_t numNeighbors = nodeMetadataPtr[currNodeID].neighborCounts[level];
            for (uint8_t i=0; i < numNeighbors; ++i) {

                const uint32_t neighborID = nodeMetadataPtr[currNodeID].neighbors[level][i];

                if (versionedSetArray[neighborID] != actualSetVersion) {
                    versionedSetArray[neighborID] = actualSetVersion;

                    float d = calculateDistanceAVX512(query_emb, &embeddingsView[neighborID, 0]);

                    if (buffer_size < ef || d < buffer[buffer_size - 1].dist) {
                        uint16_t insert_pos = (buffer_size < ef) ? buffer_size : (ef - 1);
                        while (insert_pos > 0 && buffer[insert_pos - 1].dist > d) {
                            buffer[insert_pos] = buffer[insert_pos - 1];
                            insert_pos--;
                        }
                        buffer[insert_pos] = {neighborID, d, false};

                        if (insert_pos <= curr_idx) {
                            curr_idx = insert_pos - 1;
                        }

                        if (buffer_size < ef) {
                            buffer_size++;
                        }
                    }
                }
            }
        }
        curr_idx++;
    }

    if (actualSetVersion == std::numeric_limits<uint16_t>::max()) {
        std::ranges::fill(versionedSetArray, 0);
        actualSetVersion = 0;
    } else {
        actualSetVersion++;
    }

    for (uint16_t i = 0; i < buffer_size; ++i) {
        founded_neighbors[i] = buffer[i].id;
    }
    return buffer_size;
}

std::vector<uint64_t> HNSW::search(float* query_emb, const uint16_t ef, const uint16_t k) {
    uint32_t actualEntryPoint = entryPointID;

    normalizeVectorAVX512(query_emb);

    uint32_t founded_neighbors[ef];
    for (int i = maxLevel; i >= 1; --i) {
        if (!searchLayer(query_emb, actualEntryPoint, 1, i, founded_neighbors)) {
            actualEntryPoint = founded_neighbors[0];
        }
    }

    const uint16_t numNeighbours = searchLayer(query_emb, actualEntryPoint, ef, 0, founded_neighbors);
    std::vector<uint64_t> result;
    const uint16_t actualK = std::min(k, numNeighbours);
    result.reserve(actualK);
    for (uint16_t i = 0; i < actualK; i++) {
        result.push_back(nodeMetadataPtr[founded_neighbors[i]].externalId);
    }
    return result;
}