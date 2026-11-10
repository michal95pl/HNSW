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
#include <complex>
#include <utils.h>

class HNSW {
    uint32_t entryPointID = 0;
    uint8_t actualMaxLevel = 0;

    static constexpr uint8_t HNSW_MAX_LEVEL = 6;
    static constexpr uint8_t MAX_EF = 200;

    static constexpr uint16_t EMB_DIM = 768;
    static constexpr uint32_t HNSW_SIZE = 100000;
    static constexpr uint8_t M = 16;
    const float ml = 1.0 / std::log(M);

    uint32_t embeddingsSize = 0;
    std::unique_ptr<uint64_t[]> externalIdsPtr;
    std::unique_ptr<uint8_t[]> upper_neighbors_countsPtr;
    std::unique_ptr<uint32_t[]> upper_neighborsPtr;
    std::unique_ptr<uint8_t[]> layer0_countsPtr;
    std::unique_ptr<uint32_t[]> layer0_neighborsPtr;
    AlignedUniquePtr<float> embeddingsPtr;


    std::mdspan<uint32_t, std::extents<size_t, HNSW_SIZE, M * 2>> layer0_neighborsView;

    std::mdspan<uint32_t, std::extents<size_t, HNSW_SIZE, HNSW_MAX_LEVEL, M>> upper_layer_neighborsView;
    std::mdspan<uint8_t, std::extents<size_t, HNSW_SIZE, HNSW_MAX_LEVEL>> upper_neighbors_countsView;
    std::mdspan<float, std::extents<size_t, HNSW_SIZE, EMB_DIM>> embeddingsView;


    uint16_t actualSetVersion = 1;
    std::vector<uint16_t> versionedSetArray;

    struct BufferElement {
        uint32_t id;
        float dist;
        constexpr bool operator<(const BufferElement& other) const noexcept {
            return dist < other.dist;
        }
    };

    struct SearchBufferElement : BufferElement {
        bool expanded;
    };

    [[nodiscard]] uint16_t select_neighbors_heuristic(uint32_t baseNodeID, std::span<uint32_t> candidates, std::span<uint32_t> foundedNeighbors, uint16_t m, bool keepPrunedConnections) const;

    [[nodiscard]] uint8_t getRandomLevel() const;

    template<bool IsLayer0>
    uint16_t searchLayer(const float* query_emb, uint32_t entry_point, uint16_t ef, uint8_t level, uint32_t* founded_neighbors);


public:

    explicit HNSW();

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
