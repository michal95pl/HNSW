//
// Created by michal on 3.04.2026.
//

#ifndef HNSW_HNSW_H
#define HNSW_HNSW_H

#include <vector>
#include <cstdint>
#include <aligned_utils.h>
#include <memory>
#include <mdspan>
#include <immintrin.h>
#include <algorithm>

class HNSW {

    uint32_t entryPointID = 0;
    uint8_t actualMaxLevel = 0;

    static constexpr uint16_t EMB_DIM = 768;
    static constexpr uint8_t HNSW_MAX_LEVEL = 10;
    static constexpr uint8_t HNSW_MAX_NEIGHBORS = 30;
    static constexpr uint32_t HNSW_SIZE = 100000;
    double ml;

    uint32_t embeddingsSize = 0;
    std::unique_ptr<uint64_t[]> externalIdsPtr;
    std::unique_ptr<uint8_t[]> upper_neighbors_countsPtr;
    std::unique_ptr<uint32_t[]> upper_neighborsPtr;
    std::unique_ptr<uint8_t[]> layer0_countsPtr;
    std::unique_ptr<uint32_t[]> layer0_neighborsPtr;
    AlignedUniquePtr<float> embeddingsPtr;

    std::mdspan<uint32_t, std::extents<size_t, HNSW_SIZE, HNSW_MAX_NEIGHBORS+40>> layer0_neighborsView;
    std::mdspan<uint32_t, std::extents<size_t, HNSW_SIZE, HNSW_MAX_LEVEL - 1, HNSW_MAX_NEIGHBORS + 40>> upper_layer_neighborsView;
    std::mdspan<uint8_t, std::extents<size_t, HNSW_SIZE, HNSW_MAX_LEVEL - 1>> upper_neighbors_countsView;
    std::mdspan<float, std::extents<size_t, HNSW_SIZE, EMB_DIM>> embeddingsView;


    uint16_t actualSetVersion = 1;
    std::vector<uint16_t> versionedSetArray;

    struct BufferElement {
        uint32_t id;
        float dist;
        bool expanded;
    };


    [[nodiscard]] uint8_t getRandomLevel() const;

    template<bool IsLayer0>
    uint16_t searchLayer(const float* query_emb, const uint32_t entry_point, const uint16_t ef, const uint8_t level, uint32_t* founded_neighbors) {

        std::array<BufferElement, 201> buffer{};
        uint16_t buffer_size = 0;

        const float dist = calculateDistanceAVX512(query_emb, &embeddingsView[entry_point, 0]);

        buffer[buffer_size++] = {entry_point, dist, false};
        versionedSetArray[entry_point] = actualSetVersion;

        uint32_t curr_idx = 0;
        while (curr_idx < buffer_size) {

            if (buffer_size >= ef && buffer[curr_idx].dist > buffer[buffer_size - 1].dist) break;

            if (curr_idx + 1 < buffer_size) {
                const uint32_t nextNodeID = buffer[curr_idx + 1].id;
                if constexpr (IsLayer0) {
                    _mm_prefetch(&layer0_countsPtr[nextNodeID], _MM_HINT_T0);
                } else {
                    _mm_prefetch(&upper_neighbors_countsView[nextNodeID, level - 1], _MM_HINT_T0);
                }
            }

            if (!buffer[curr_idx].expanded) {
                buffer[curr_idx].expanded = true;
                const uint32_t currNodeID = buffer[curr_idx].id;

                const uint8_t numNeighbors = IsLayer0
                ? layer0_countsPtr[currNodeID]
                : upper_neighbors_countsView[currNodeID, level - 1];

                const uint32_t* current_neighbors = nullptr;
                if constexpr (IsLayer0) {
                    current_neighbors = &layer0_neighborsView[currNodeID, 0];
                } else {
                    current_neighbors = &upper_layer_neighborsView[currNodeID, level - 1, 0];
                }

                for (uint8_t i=0; i < numNeighbors; ++i) {

                    if (i + 1 < numNeighbors) {
                        const uint32_t nextID = current_neighbors[i + 1];

                        _mm_prefetch(&embeddingsView[nextID, 0], _MM_HINT_T0);
                        _mm_prefetch(&embeddingsView[nextID, 16], _MM_HINT_T0);
                        _mm_prefetch(&versionedSetArray[nextID], _MM_HINT_T0);
                    }

                    const uint32_t neighborID = current_neighbors[i];

                    if (versionedSetArray[neighborID] != actualSetVersion) {
                        versionedSetArray[neighborID] = actualSetVersion;

                        const float d = calculateDistanceAVX512(query_emb, &embeddingsView[neighborID, 0]);

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
            actualSetVersion = 1;
        } else {
            actualSetVersion++;
        }

        for (uint16_t i = 0; i < buffer_size; ++i) {
            founded_neighbors[i] = buffer[i].id;
        }
        return buffer_size;
    }

    template<bool IsLayer0>
    void shrinkNeighbors(const uint32_t neighbor_id, const uint8_t layer, const uint16_t maxNeighbors) {

        const uint8_t neighborCount = IsLayer0
        ? layer0_countsPtr[neighbor_id]
        : upper_neighbors_countsView[neighbor_id, layer-1];

        if (neighborCount <= maxNeighbors) {
            return;
        }


        std::pair<float, uint32_t> dist_pairs[neighborCount];

        const uint32_t* current_neighbors = nullptr;
        if constexpr (IsLayer0) {
            current_neighbors = &layer0_neighborsView[neighbor_id, 0];
        } else {
            current_neighbors = &upper_layer_neighborsView[neighbor_id, layer - 1, 0];
        }

        for (uint16_t i=0; i < neighborCount; ++i) {

            if (i + 1 < neighborCount) {
                const uint32_t nextNeighborId = current_neighbors[i + 1];
                _mm_prefetch(&embeddingsView[nextNeighborId, 0], _MM_HINT_T0);
                _mm_prefetch(&embeddingsView[nextNeighborId, 16], _MM_HINT_T0);
            }

            const uint32_t neighborId = current_neighbors[i];

            dist_pairs[i].second = neighborId;
            dist_pairs[i].first = calculateDistanceAVX512(&embeddingsView[neighbor_id, 0], &embeddingsView[neighborId, 0]);
        }

        auto data_view = std::span(dist_pairs, neighborCount);
        std::ranges::nth_element(data_view, data_view.begin() + maxNeighbors, std::less{}, &std::pair<float, uint32_t>::first);

        (IsLayer0 ? layer0_countsPtr[neighbor_id] : upper_neighbors_countsView[neighbor_id, layer-1]) = maxNeighbors;

        for (uint16_t i = 0; i < maxNeighbors; ++i) {
            (IsLayer0 ? layer0_neighborsView[neighbor_id, i] : upper_layer_neighborsView[neighbor_id, layer-1, i]) = dist_pairs[i].second;
        }
    }

public:
    /**
     * @param M max number of neighbors per layer (except layer 0)
     */
    explicit HNSW(uint8_t M);

    /**
     * @param embedding1 should be normalized
     * @param embedding2 should be normalized
     * @return cosine distance between embedding1 and embedding2.
     */
    static float calculateDistanceAVX512(const float* vec1, const float* vec2);
    static float calculateDistanceAVX512(const float* vec1, const float* vec2, const float* next_vec2);
    static float calculateDistance(const std::vector<float>& vec1, const std::vector<float>& vec2);
    [[nodiscard]] std::vector<uint64_t> search(float* query_emb, uint16_t ef, uint16_t k);
    void addPoint(uint64_t externalID, const float* embedding, uint16_t ef=10);
    static void normalizeVectorAVX512(float* vec);
    static void normalizeVector(std::vector<float>& vec);
};

#endif
