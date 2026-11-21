//
// Created by michal on 13.04.2026.
//

#include <../include/hnsw/progress_bar.h>
#include <iostream>
#include <iomanip>
#include <chrono>

ProgressBar::ProgressBar(const std::size_t total, const uint16_t barWidth, const std::string &description)
: total(total), barWidth(barWidth),description(description), etaStartTime(std::chrono::steady_clock::now()) {}

void ProgressBar::update(const std::size_t current) const {
    if (total == 0) return;

    float progress = static_cast<float>(current) / total;
    if (progress > 1.0f) progress = 1.0f;

    const uint16_t pos = static_cast<uint16_t>(barWidth * progress);

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - etaStartTime).count();

    uint32_t eta = 0;
    if (current > 0) {
        eta = static_cast<uint32_t>(static_cast<double>(elapsed) / current * (total - current));
    }
    const uint32_t mins = eta / 60;
    const uint16_t secs = eta % 60;

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
              << "ETA: " << std::setfill('0') << std::setw(3) << mins << ":"
              << std::setfill('0') << std::setw(2) << secs << " [" << description << "]  " << std::flush;
}

void ProgressBar::start_timer() {
    if (!timerPaused) return;
    timerStartTime = std::chrono::steady_clock::now();
    timerPaused = false;
}
void ProgressBar::stop_timer() {
    if (timerPaused) return;
    timerPaused = true;
    accumulatedTime += (std::chrono::steady_clock::now() - timerStartTime);
    timerStartTime = std::chrono::steady_clock::time_point{};
}

void ProgressBar::finish() {
    if (finished) return;

    std::cout << std::endl;

    stop_timer();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(accumulatedTime).count();
    std::cout << "Total time: " << elapsed / 60 << " mins " << elapsed % 60 << " secs" << std::endl;

    std::cout << std::defaultfloat << std::setprecision(6);
    finished = true;
}

ProgressBar::~ProgressBar() {
    finish();
}
