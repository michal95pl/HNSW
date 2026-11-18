//
// Created by michal on 3.04.2026.
//

#pragma once

#include <string>
#include <hnsw/HNSW.h>

static constexpr uint32_t batchSize = 1000;

void loadVectorsToHNSW(HNSW &hnsw, const std::string &file);

float getRecall(HNSW &hnsw, const std::string &groundTruthFilePath, const std::string &queryFilePath, uint8_t k);
