//
// Created by michal on 3.04.2026.
//

#include <algorithm>
#include <HNSW.h>
#include <random>
#include <queue>
#include <unordered_set>
#include <immintrin.h>

HNSW::HNSW(const uint8_t M) {
    nodes.reserve(HNSW_SIZE);
    ml = 1.0 / std::log(static_cast<double>(M));
}

uint8_t HNSW::getRandomLevel() const {
    // unsafe for threads
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    double r = dis(gen);
    if (r == 0) r = 1e-10;

    auto level = static_cast<uint8_t>(-std::log(r) * ml);
    if (level > HNSW_MAX_LEVEL) level = HNSW_MAX_LEVEL;
    return level;
}

void HNSW::shrinkNeighbors(const uint32_t neighbor_id, const uint8_t layer, const uint16_t maxNeighbors) {

    if (nodes[neighbor_id].layers[layer].size() <= maxNeighbors) {
        return;
    }

    auto& neighbors = nodes[neighbor_id].layers[layer];

    std::vector<std::pair<float, uint32_t>> dist_pairs;

    for (uint32_t id : neighbors) {
        dist_pairs.emplace_back(calculateDistanceAVX512(nodes[neighbor_id].embedding, nodes[id].embedding), id);
    }

    std::ranges::nth_element(dist_pairs, dist_pairs.begin() + maxNeighbors);

    neighbors.clear();
    for (uint16_t i = 0; i < maxNeighbors; ++i) {
        neighbors.push_back(dist_pairs[i].second);
    }
}

__attribute__((target("avx512f,fma")))
void HNSW::normalizeVector(AlignedVector& vec) {
    float* data = vec.data();
    size_t size = vec.size();

    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();

    for (size_t i = 0; i < size; i += 64) {
        __m512 v1 = _mm512_load_ps(&data[i]);
        __m512 v2 = _mm512_load_ps(&data[i + 16]);
        __m512 v3 = _mm512_load_ps(&data[i + 32]);
        __m512 v4 = _mm512_load_ps(&data[i + 48]);

        sum1 = _mm512_fmadd_ps(v1, v1, sum1);
        sum2 = _mm512_fmadd_ps(v2, v2, sum2);
        sum3 = _mm512_fmadd_ps(v3, v3, sum3);
        sum4 = _mm512_fmadd_ps(v4, v4, sum4);
    }

    __m512 total_sum_v = _mm512_add_ps(_mm512_add_ps(sum1, sum2), _mm512_add_ps(sum3, sum4));
    float sum_sq = _mm512_reduce_add_ps(total_sum_v);

    if (sum_sq < 1e-12f) return;

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    __m512 inv_norm_v = _mm512_set1_ps(inv_norm);

    for (size_t i = 0; i < size; i += 64) {
        __m512 v1 = _mm512_load_ps(&data[i]);
        __m512 v2 = _mm512_load_ps(&data[i + 16]);
        __m512 v3 = _mm512_load_ps(&data[i + 32]);
        __m512 v4 = _mm512_load_ps(&data[i + 48]);

        _mm512_store_ps(&data[i],      _mm512_mul_ps(v1, inv_norm_v));
        _mm512_store_ps(&data[i + 16], _mm512_mul_ps(v2, inv_norm_v));
        _mm512_store_ps(&data[i + 32], _mm512_mul_ps(v3, inv_norm_v));
        _mm512_store_ps(&data[i + 48], _mm512_mul_ps(v4, inv_norm_v));
    }
}

void HNSW::addPoint(const uint64_t externalID, AlignedVector&& embedding, const uint16_t maxNeighbors, const uint16_t ef) {
    const uint32_t newNodeID = nodes.size();
    const uint8_t level = getRandomLevel();
    normalizeVector(embedding);

    Node newNode(externalID, std::move(embedding));
    newNode.layers.resize(level + 1);

    if (nodes.empty()) {
        nodes.push_back(std::move(newNode));
        entryPointID = 0;
        maxLevel = level;
        return;
    }

    uint32_t currObjID = entryPointID;
    nodes.push_back(std::move(newNode));

    // search best entry point for each layer above the new node's level
    for (int l = maxLevel; l > level; --l) {
        if (auto tempCandidate = searchLayer(nodes[newNodeID].embedding, currObjID, 1, l); !tempCandidate.empty()) {
            currObjID = tempCandidate[0];
        }
    }

    for (int l = std::min(level, maxLevel); l >= 0; --l) {
        auto neighbors = searchLayer(nodes[newNodeID].embedding, currObjID, ef, l);
        if (neighbors.empty()) continue;

        for (uint32_t neighborID : neighbors) {
            nodes[newNodeID].layers[l].push_back(neighborID);
            nodes[neighborID].layers[l].push_back(newNodeID);
            shrinkNeighbors(neighborID, l, 30);
        }
        currObjID = neighbors[0];
        shrinkNeighbors(newNodeID, l, maxNeighbors);
    }

    if (level > maxLevel) {
        entryPointID = newNodeID;
        maxLevel = level;
    }
}

float HNSW::calculateDistance(const AlignedVector& embedding1, const AlignedVector& embedding2) {
    float dot_product = 0;
    for (uint16_t i = 0; i < embedding1.size(); ++i) {
        dot_product += embedding1[i] * embedding2[i];
    }
    const float dist = 1.0f - dot_product;
    return (dist < 0) ? 0 : dist;
}

__attribute__((target("avx512f,fma")))
float HNSW::calculateDistanceAVX512(const AlignedVector& embedding1, const AlignedVector& embedding2) {
    const float* a = embedding1.data();
    const float* b = embedding2.data();
    size_t size = embedding1.size();

    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();

    for (size_t i = 0; i < size; i += 64) {
        sum1 = _mm512_fmadd_ps(_mm512_load_ps(&a[i]),      _mm512_load_ps(&b[i]),      sum1);
        sum2 = _mm512_fmadd_ps(_mm512_load_ps(&a[i + 16]), _mm512_load_ps(&b[i + 16]), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_load_ps(&a[i + 32]), _mm512_load_ps(&b[i + 32]), sum3);
        sum4 = _mm512_fmadd_ps(_mm512_load_ps(&a[i + 48]), _mm512_load_ps(&b[i + 48]), sum4);
    }

    __m512 total_sum = _mm512_add_ps(_mm512_add_ps(sum1, sum2), _mm512_add_ps(sum3, sum4));

    const float dot = _mm512_reduce_add_ps(total_sum);
    const float dist = 1.0f - dot;

    return (dist < 0) ? 0 : dist;
}

std::vector<uint32_t> HNSW::searchLayer(const AlignedVector& query_emb, uint32_t entry_point, const uint16_t ef, const uint8_t level) const {

    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> candidates;
    std::priority_queue<std::pair<float, uint32_t>> found_neighbors;
    std::unordered_set<uint32_t> visited;

    float dist = calculateDistanceAVX512(query_emb, nodes[entry_point].embedding);
    candidates.emplace(dist, entry_point);
    found_neighbors.emplace(dist, entry_point);
    visited.insert(entry_point);

    while (!candidates.empty()) {
        auto curr = candidates.top();
        candidates.pop();

        if (curr.first > found_neighbors.top().first) break;

        for (uint32_t neighborID : nodes[curr.second].layers[level]) {
            if (!visited.contains(neighborID)) {
                visited.insert(neighborID);

                if (float d = calculateDistanceAVX512(query_emb, nodes[neighborID].embedding); d < found_neighbors.top().first || found_neighbors.size() < ef) {
                    candidates.emplace(d, neighborID);
                    found_neighbors.emplace(d, neighborID);

                    if (found_neighbors.size() > ef) {
                        found_neighbors.pop();
                    }
                }
            }
        }
    }

    std::vector<uint32_t> result;
    const uint32_t n = found_neighbors.size();
    result.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        result[i] = found_neighbors.top().second;
        found_neighbors.pop();
    }
    return result;
}

//todo: zrobić kontroler, lub kopiowanie query
std::vector<uint64_t> HNSW::search(AlignedVector &&query_emb, const uint16_t ef, const uint16_t k) const {
    uint32_t actualEntryPoint = entryPointID;

    normalizeVector(query_emb);

    for (int i = maxLevel; i >= 1; --i) {
        const auto& temp = searchLayer(query_emb, actualEntryPoint, 1, i);
        if (!temp.empty()) {
            actualEntryPoint = temp[0];
        }
    }
    auto temp = searchLayer(query_emb, actualEntryPoint, ef, 0);
    std::vector<uint64_t> result;
    const uint16_t actualK = std::min(k, static_cast<uint16_t>(temp.size()));
    result.reserve(actualK);
    for (uint16_t i = 0; i < actualK; i++) {
        result.push_back(nodes[temp[i]].externalId);
    }
    return result;
}