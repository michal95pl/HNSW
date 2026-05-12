//
// Created by michal on 11.05.2026.
//

#ifndef HNSW_ALIGNEDUTILS_H
#define HNSW_ALIGNEDUTILS_H

#include <new>
#include <bit>
#include <cstdlib>
#include <vector>

template <typename T, std::size_t Alignment>
    requires (Alignment >= alignof(T)) && (std::has_single_bit(Alignment))
    struct AlignedAllocator {
    using value_type = T;

    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    AlignedAllocator() noexcept = default;

    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(const std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t size = (n * sizeof(T) + Alignment - 1) & ~(Alignment - 1);

        void* ptr = std::aligned_alloc(Alignment, size);

        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

using AlignedVector = std::vector<float, AlignedAllocator<float, 64>>;

#endif
