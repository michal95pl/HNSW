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

class HNSW {

    uint32_t entryPointID = 0;
    uint8_t maxLevel = 0;

    static constexpr uint16_t EMB_DIM = 768;
    static constexpr uint8_t HNSW_MAX_LEVEL = 10;
    static constexpr uint8_t HNSW_MAX_NEIGHBORS = 30;
    static constexpr uint32_t HNSW_SIZE = 100000;
    double ml;

    struct alignas(64) NodeMetadata {
        uint32_t externalId{};
        uint8_t neighborCounts[HNSW_MAX_LEVEL]{};
        uint32_t neighbors[HNSW_MAX_LEVEL][HNSW_MAX_NEIGHBORS+40]{};
        uint8_t layer{};
    };

    uint32_t embeddingsSize = 0;
    AlignedUniquePtr<float> embeddingsPtr;
    std::unique_ptr<NodeMetadata[]> nodeMetadataPtr;
    std::mdspan<float, std::extents<size_t, std::dynamic_extent, EMB_DIM>> embeddingsView;

    /**
     * Shrinks the neighbors of a given node at a specific layer to a maximum number of neighbors.
     * @param neighbor_id
     * @param layer
     * @param maxNeighbors
     */
    void shrinkNeighbors(uint32_t neighbor_id, uint8_t layer, uint16_t maxNeighbors);
    [[nodiscard]] uint8_t getRandomLevel() const;
    [[nodiscard]] uint16_t searchLayer(const float* query_emb, uint32_t entry_point, uint16_t ef, uint8_t level, uint32_t* founded_neighbors) const;

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
    static float calculateDistance(const std::vector<float>& vec1, const std::vector<float>& vec2);
    [[nodiscard]] std::vector<uint64_t> search(float* query_emb, uint16_t ef, uint16_t k) const;
    void addPoint(uint64_t externalID, const float* embedding, uint16_t ef=10);
    static void normalizeVectorAVX512(float* vec);
    static void normalizeVector(std::vector<float>& vec);
};

#endif
