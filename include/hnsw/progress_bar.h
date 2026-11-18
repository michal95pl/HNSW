//
// Created by michal on 13.04.2026.
//

#ifndef HNSW_PROGRESS_BAR_H
#define HNSW_PROGRESS_BAR_H

#include <chrono>

class ProgressBar {

    std::size_t total;
    uint16_t barWidth = 50;
    bool finished = false;
    std::chrono::steady_clock::time_point startTime;
    std::string description;

public:
    explicit ProgressBar(std::size_t total, uint16_t barWidth, const std::string &description);
    ~ProgressBar();

    void update(std::size_t current) const;
    void finish();
};

#endif