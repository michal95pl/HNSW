//
// Created by michal on 28.10.2026.
//

#ifndef HNSW_UTILS_H
#define HNSW_UTILS_H
#include <cstdint>
#include <span>
#include <functional>

template <typename T, typename Compare = std::less<T>>
inline uint16_t insertIntoBuffer(const T value, const std::span<T> buffer, uint16_t& current_buffer_size, const uint16_t max_buffer_size, const Compare comp = Compare{}) noexcept {

    if (current_buffer_size == max_buffer_size && !comp(value, buffer[max_buffer_size - 1])) {
        return max_buffer_size;
    }

    uint16_t insert_pos = (current_buffer_size < max_buffer_size)
                          ? current_buffer_size
                          : (max_buffer_size - 1);


    while (insert_pos > 0 && comp(value, buffer[insert_pos - 1])) {
        buffer[insert_pos] = std::move(buffer[insert_pos - 1]);
        --insert_pos;
    }

    buffer[insert_pos] = std::move(value);

    if (current_buffer_size < max_buffer_size) {
        ++current_buffer_size;
    }

    return insert_pos;
}

#endif //HNSW_UTILS_H
