//
// Created by michal on 3.04.2026.
//

#ifndef HNSW_HNSW_H
#define HNSW_HNSW_H

#include <vector>
#include <cstdint>
#include <aligned_utils.h>

class HNSW {

    struct Node {
        uint64_t externalId;
        AlignedVector embedding;
        std::vector<std::vector<uint32_t>> layers;

        explicit Node(const uint64_t _eID, AlignedVector&& _emb)
            : externalId(_eID), embedding(std::move(_emb)) {
        }
    };

    std::vector<Node> nodes;
    uint32_t entryPointID = 0;
    uint8_t maxLevel = 0;
    static constexpr uint8_t HNSW_MAX_LEVEL = 10;
    static constexpr uint32_t HNSW_SIZE = 100000;
    double ml;

    /**
     * Shrinks the neighbors of a given node at a specific layer to a maximum number of neighbors.
     * @param neighbor_id
     * @param layer
     * @param maxNeighbors
     */
    void shrinkNeighbors(uint32_t neighbor_id, uint8_t layer, uint16_t maxNeighbors);
    [[nodiscard]] uint8_t getRandomLevel() const;
    [[nodiscard]] std::vector<uint32_t> searchLayer(const AlignedVector& query_emb, uint32_t entry_point, uint16_t ef, uint8_t level) const;

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
    [[nodiscard]] static float calculateDistance(const AlignedVector& embedding1, const AlignedVector& embedding2);
    static float calculateDistanceAVX512(const AlignedVector& embedding1, const AlignedVector& embedding2);
    [[nodiscard]] std::vector<uint64_t> search(AlignedVector &&query_emb, uint16_t ef, uint16_t k) const;
    void addPoint(uint64_t externalID, AlignedVector&& embedding, uint16_t maxNeighbors, uint16_t ef=10);
    static void normalizeVector(AlignedVector& vec);
};

#endif
