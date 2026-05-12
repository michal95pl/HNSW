//
// Created by michal on 14.04.2026.
//

#ifndef HNSW_GROUND_TRUTH_FILE_HANDLER_H
#define HNSW_GROUND_TRUTH_FILE_HANDLER_H
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#pragma pack(push, 1)
/**
 * @brief Represents the distance from a query vector to a specific vector in the dataset, identified by its ID.
 */
struct GroundTruthDistance {
    uint64_t id;
    float distance;

    bool operator<(const GroundTruthDistance& other) const {
        return distance < other.distance;
    }
};
#pragma pack(pop)

class GroundTruthFileHandler {
    enum class State {Closed, Reading, Writing};
    State currentState_ = State::Closed;

    std::fstream file_;
    std::string filePath_;
    uint16_t k_;
    bool ensureState(State required);
    void closeFile();
    bool loadMetadata();
public:
    explicit GroundTruthFileHandler(std::string path);
    ~GroundTruthFileHandler();

    /**
     * Creates or overwrite file
     * @param k number of nearest neighbors to store for each query vector.
     * @return  true if file was created
     */
    bool createFile(uint16_t k);

    /**
     * Appends a vector to the file. The size of the vector must be equal to k specified in createFile.
     * The vector should be ordered as in the query file,meaning that the first vector corresponds to the first
     * query vector, the second to the second, and so on.
     * @param vector
     * @return true if vector was successfully appended
     */
    bool appendQueryVectorDistances(const std::vector<GroundTruthDistance> &vector);

    /**
     *
     * @param queryId index of the query vector in the file (starting from 0). This also represents the id in the query vector file
     * @param outVector
     * @return true if vector was successfully read
     */
    bool getVector(uint32_t queryId, std::vector<GroundTruthDistance> &outVector);

    /**
     *
     * @return number of vectors in the file
     */
    uint32_t getNumVectors() const;

    /**
     *
     * @return all vectors in the file. The order of vectors corresponds to the order of query vectors in the query vector file.
     */
    std::vector<GroundTruthDistance> getAllVectors();

    uint16_t getK() const { return k_; }
};

#endif
