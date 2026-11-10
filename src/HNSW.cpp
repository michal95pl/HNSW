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

HNSW::HNSW() :
    externalIdsPtr(std::make_unique<uint64_t[]>(HNSW_SIZE)),
    upper_neighbors_countsPtr(std::make_unique<uint8_t[]>(HNSW_SIZE * HNSW_MAX_LEVEL)),
    upper_neighborsPtr(std::make_unique<uint32_t[]>(HNSW_MAX_LEVEL * HNSW_SIZE * M)),
    layer0_countsPtr(std::make_unique<uint8_t[]>(HNSW_SIZE)),
    layer0_neighborsPtr(std::make_unique<uint32_t[]>(HNSW_SIZE * M * 2)),
    embeddingsPtr(make_aligned_array<float>(HNSW_SIZE * EMB_DIM)),
    layer0_neighborsView(layer0_neighborsPtr.get()),
    upper_layer_neighborsView(upper_neighborsPtr.get()),
    upper_neighbors_countsView(upper_neighbors_countsPtr.get()),
    embeddingsView(embeddingsPtr.get()),
    versionedSetArray(HNSW_SIZE, 0){}

uint8_t HNSW::getRandomLevel() const {
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

    if (embeddingsSize == 1) {
        entryPointID = 0;
        actualMaxLevel = level;
        return;
    }

    uint32_t currNodeID = entryPointID;
    uint32_t search_results[MAX_EF];

    for (int l = actualMaxLevel; l > level; --l) {
        if (searchLayer<false>(&embeddingsView[newNodeID, 0], currNodeID, 1, l, search_results)) {
            currNodeID = search_results[0];
        }
    }

    for (int l = std::min<int>(level, actualMaxLevel); l >= 0; --l) {
        const bool isLayer0 = (l == 0);
        const uint16_t maxM = isLayer0 ? (M * 2) : M;

        uint16_t num_candidates = 0;
        if (isLayer0) {
            num_candidates = searchLayer<true>(&embeddingsView[newNodeID, 0], currNodeID, ef, 0, search_results);
        } else {
            num_candidates = searchLayer<false>(&embeddingsView[newNodeID, 0], currNodeID, ef, l, search_results);
        }
        if (num_candidates == 0) continue;

        uint32_t* newNodeConnections = isLayer0
            ? &layer0_neighborsView[newNodeID, 0]
            : &upper_layer_neighborsView[newNodeID, l - 1, 0];

        const uint16_t selected_count = select_neighbors_heuristic(
            newNodeID,
            std::span(search_results, num_candidates),
            std::span(newNodeConnections, maxM),
            maxM,
            true
        );

        if (isLayer0) {
            layer0_countsPtr[newNodeID] = selected_count;
        } else {
            upper_neighbors_countsView[newNodeID, l - 1] = selected_count;
        }

        for (uint16_t i = 0; i < selected_count; ++i) {
            const uint32_t neighborID = newNodeConnections[i];

            uint8_t& neighborCount = isLayer0
                ? layer0_countsPtr[neighborID]
                : upper_neighbors_countsView[neighborID, l - 1];

            uint32_t* neighborConnections = isLayer0
                ? &layer0_neighborsView[neighborID, 0]
                : &upper_layer_neighborsView[neighborID, l - 1, 0];

            if (neighborCount < maxM) {
                neighborConnections[neighborCount++] = newNodeID;
            } else {
                std::vector neighbor_candidates(neighborConnections, neighborConnections + neighborCount);
                neighbor_candidates.push_back(newNodeID);

                const uint16_t new_neighbor_count = select_neighbors_heuristic(
                    neighborID,
                    std::span(neighbor_candidates.data(), neighbor_candidates.size()),
                    std::span(neighborConnections, maxM),
                    maxM,
                    true
                );
                neighborCount = new_neighbor_count;
            }
        }

        if (l > 0 && num_candidates > 0) {
            currNodeID = search_results[0];
        }
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

    uint32_t founded_neighbors[MAX_EF];
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

uint16_t HNSW::select_neighbors_heuristic(const uint32_t baseNodeID, const std::span<uint32_t> candidates, const std::span<uint32_t> foundedNeighbors, const uint16_t m, const bool keepPrunedConnections) const {
    std::array<BufferElement, MAX_EF> candidates_buffer{};
    uint16_t candidates_buffer_size = 0;

    std::array<BufferElement, M*2> discarded_buffer{};
    uint16_t discarded_buffer_size = 0;

    for (const uint32_t candidateID : candidates) {
        const float dist = calculateDistanceAVX512(&embeddingsView[baseNodeID, 0], &embeddingsView[candidateID, 0]);
        insertIntoBuffer<BufferElement>(
            BufferElement{candidateID, dist},
            candidates_buffer,
            candidates_buffer_size,
            MAX_EF
        );
    }

    uint16_t result_size = 0;
    for (uint16_t i = 0; i < candidates_buffer_size && result_size < m; ++i) {
        const BufferElement& candidate = candidates_buffer[i];

        bool is_closer_than_any = true;

        for (uint16_t j = 0; j < result_size; ++j) {
            const uint32_t accepted_neighbor = foundedNeighbors[j];
            const float dist = calculateDistanceAVX512(&embeddingsView[candidate.id, 0], &embeddingsView[accepted_neighbor, 0]);
            if (dist < candidate.dist) {
                is_closer_than_any = false;
                break;
            }
        }

        if (is_closer_than_any) {
            foundedNeighbors[result_size++] = candidate.id;
        } else {
            discarded_buffer[discarded_buffer_size++] = candidate;
        }
    }

    if (keepPrunedConnections && result_size < m) {
        for (uint16_t i = 0; i < discarded_buffer_size && result_size < M; ++i) {
            foundedNeighbors[result_size++] = discarded_buffer[i].id;
        }
    }
    return result_size;
}

template<bool IsLayer0>
uint16_t HNSW::searchLayer(const float* query_emb, const uint32_t entry_point, const uint16_t ef, const uint8_t level, uint32_t* founded_neighbors) {

    // buffer on stack, is sorted. Last element is the farthest.
    std::array<SearchBufferElement, MAX_EF> buffer{};
    uint16_t buffer_size = 0;

    const float dist = calculateDistanceAVX512(query_emb, &embeddingsView[entry_point, 0]);

    buffer[buffer_size++] = {entry_point, dist, false};
    versionedSetArray[entry_point] = actualSetVersion;


    while (true) {
        int32_t curr_idx = -1;
        for (uint16_t i = 0; i < buffer_size; ++i) {
            if (!buffer[i].expanded) {
                curr_idx = i;
                break;
            }
        }
        if (curr_idx == -1) break;

        if (buffer_size >= ef && buffer[curr_idx].dist > buffer[buffer_size - 1].dist) {
            break;
        }

        buffer[curr_idx].expanded = true;
        const uint32_t currNodeID = buffer[curr_idx].id;

        for (uint16_t next = curr_idx + 1; next < buffer_size; ++next) {
            if (!buffer[next].expanded) {
                const uint32_t nextID = buffer[next].id;
                if constexpr (IsLayer0) {
                    _mm_prefetch(&layer0_countsPtr[nextID], _MM_HINT_T0);
                } else {
                    _mm_prefetch(&upper_neighbors_countsView[nextID, level - 1], _MM_HINT_T0);
                }
                break;
            }
        }

        const uint8_t numNeighbors = IsLayer0
            ? layer0_countsPtr[currNodeID]
            : upper_neighbors_countsView[currNodeID, level - 1];

        const uint32_t* current_neighbors = nullptr;
        if constexpr (IsLayer0) {
            current_neighbors = &layer0_neighborsView[currNodeID, 0];
        } else {
            current_neighbors = &upper_layer_neighborsView[currNodeID, level - 1, 0];
        }

        for (uint8_t i = 0; i < numNeighbors; ++i) {
            if (i + 1 < numNeighbors) {
                const uint32_t nextID = current_neighbors[i + 1];
                _mm_prefetch(&embeddingsView[nextID, 0], _MM_HINT_T0);
                _mm_prefetch(&embeddingsView[nextID, 16], _MM_HINT_T0);
                _mm_prefetch(&versionedSetArray[nextID], _MM_HINT_T0);
            }

            const uint32_t neighborID = current_neighbors[i];

            if (versionedSetArray[neighborID] == actualSetVersion) continue;
            versionedSetArray[neighborID] = actualSetVersion;

            const float d = calculateDistanceAVX512(query_emb, &embeddingsView[neighborID, 0]);

            insertIntoBuffer<SearchBufferElement>(
                SearchBufferElement{neighborID, d, false},
                buffer,
                buffer_size,
                ef
            );
        }
    }

    if (actualSetVersion == std::numeric_limits<uint16_t>::max()) {
        std::ranges::fill(versionedSetArray, 0);
        actualSetVersion = 1;
    } else {
        actualSetVersion++;
    }

    for (uint16_t i = 0; i < buffer_size; ++i) {
        founded_neighbors[i] = buffer[i].id;
    }
    return buffer_size;
}

template uint16_t HNSW::searchLayer<true>(
    const float*, uint32_t, uint16_t, uint8_t, uint32_t*);

template uint16_t HNSW::searchLayer<false>(
    const float*, uint32_t, uint16_t, uint8_t, uint32_t*);