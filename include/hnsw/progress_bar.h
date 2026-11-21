//
// Created by michal on 13.04.2026.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <chrono>

class ProgressBar {

    std::size_t total;
    uint16_t barWidth;
    bool finished = false;
    std::string description;

    // timer
    bool timerPaused = true;
    std::chrono::steady_clock::duration accumulatedTime{0};
    std::chrono::steady_clock::time_point timerStartTime;

    //eta
    std::chrono::steady_clock::time_point etaStartTime;

public:
    explicit ProgressBar(std::size_t total, uint16_t barWidth, const std::string &description);
    ~ProgressBar();

    void update(std::size_t current) const;
    void finish();

    void start_timer();
    void stop_timer();
};