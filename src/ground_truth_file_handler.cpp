//
// Created by michal on 14.04.2026.
//

#include "ground_truth_file_handler.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <utility>

GroundTruthFileHandler::GroundTruthFileHandler(std::string path) : filePath_(std::move(path)), k_(0) {
    loadMetadata();
}

bool GroundTruthFileHandler::ensureState(const State required) {
    if (currentState_ == required && (required == State::Closed || file_.is_open())) {
        return true;
    }

    if (required != State::Closed && !std::filesystem::exists(filePath_)) {
        currentState_ = State::Closed;
        return false;
    }

    closeFile();

    switch (required) {
        case State::Closed:
            break;
        case State::Reading:
            file_.open(filePath_, std::ios::binary | std::ios::in);
            break;
        case State::Writing:
            file_.open(filePath_, std::ios::binary | std::ios::out | std::ios::app);
            break;
    }
    if (!file_.is_open() && required != State::Closed) {
        currentState_ = State::Closed;
        return false;
    }
    currentState_ = required;
    return true;
}

bool GroundTruthFileHandler::loadMetadata() {
    if (!ensureState(State::Reading)) return false;

    uint16_t file_k;
    file_.seekg(0, std::ios::beg);
    file_.read(reinterpret_cast<char*>(&file_k), sizeof(uint16_t));

    if (!file_.good() || file_k == 0) return false;

    k_ = file_k;
    return true;
}

void GroundTruthFileHandler::closeFile() {
    if (file_.is_open()) {
        file_.close();
    }
    file_.clear();
}

GroundTruthFileHandler::~GroundTruthFileHandler() {
    closeFile();
}

bool GroundTruthFileHandler::createFile(const uint16_t k) {
    closeFile();

    file_.open(filePath_, std::ios::binary | std::ios::out);
    if (!file_.is_open()) {
        currentState_ = State::Closed;
        return false;
    }

    file_.write(reinterpret_cast<const char*>(&k), sizeof(uint16_t));
    if (file_.good()) {
        k_ = k;
        currentState_ = State::Writing;
        return true;
    }
    currentState_ = State::Closed;
    return false;
}

bool GroundTruthFileHandler::appendQueryVectorDistances(const std::vector<GroundTruthDistance> &vector) {
    if (!ensureState(State::Writing)) return false;

    if (vector.size() != k_) {
        return false;
    }

    file_.write(reinterpret_cast<const char*>(vector.data()), vector.size() * sizeof(GroundTruthDistance));

    return file_.good();
}

uint32_t GroundTruthFileHandler::getNumVectors() const {
    if (k_ == 0) return 0;

    const uintmax_t size = std::filesystem::file_size(filePath_);
    return (size - sizeof(uint16_t)) / (k_ * sizeof(GroundTruthDistance));
}

bool GroundTruthFileHandler::getVector(const uint32_t queryId, std::vector<GroundTruthDistance> &outVector) {
    if (!ensureState(State::Reading)) return false;

    if (k_ == 0) {return false;}

    const std::streamoff offset = sizeof(uint16_t) + (static_cast<std::streamoff>(queryId) * k_ * sizeof(GroundTruthDistance));
    file_.seekg(offset, std::ios::beg);
    outVector.resize(k_);
    file_.read(reinterpret_cast<char*>(outVector.data()), static_cast<std::streamsize>(k_ * sizeof(GroundTruthDistance)));

    return file_.good();
}

std::vector<GroundTruthDistance> GroundTruthFileHandler::getAllVectors() {
    if (!ensureState(State::Reading) || k_ == 0) return {};

    uint32_t numVectors = getNumVectors();
    std::vector<GroundTruthDistance> allDistances(numVectors * k_);

    file_.seekg(sizeof(uint16_t), std::ios::beg);

    file_.read(reinterpret_cast<char*>(allDistances.data()), static_cast<std::streamsize>(allDistances.size() * sizeof(GroundTruthDistance)));

    if (!file_.good()) {
        return {};
    }

    return allDistances;
}
