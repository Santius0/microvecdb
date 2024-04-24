#ifndef MICROVECDB_ALIGNED_VECTOR_H
#define MICROVECDB_ALIGNED_VECTOR_H

#include <cstdlib>
#include <memory>
#include <vector>

namespace mvdb {

    #if defined(__AVX2__) || defined(__AVX__) // if using avx will be using __mm256 so 32-bytes
    constexpr std::size_t alignment = 32;
    #elif defined(__ARM_NEON)                // if using neon will be using float32x4 so 16-bytes
    constexpr std::size_t alignment = 16;
    #endif

    template <typename T, std::size_t Alignment>
    class AlignedAllocator {
    public:
        using value_type = T;

        AlignedAllocator() = default;

        template <typename U>
        constexpr explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        [[nodiscard]] T* allocate(std::size_t n) {
            if (auto p = static_cast<T*>(aligned_alloc(Alignment, n * sizeof(T)))) {
                return p;
            }
            throw std::bad_alloc();
        }

        void deallocate(T* p, std::size_t) noexcept {
            free(p);
        }
    };

    template <typename T, std::size_t TAlignment, typename U, std::size_t UAlignment>
    bool operator==(const AlignedAllocator<T, TAlignment>&, const AlignedAllocator<U, UAlignment>&) noexcept {
        return TAlignment == UAlignment;
    }

    template <typename T, std::size_t TAlignment, typename U, std::size_t UAlignment>
    bool operator!=(const AlignedAllocator<T, TAlignment>& a, const AlignedAllocator<U, UAlignment>& b) noexcept {
        return !(a == b);
    }

//    template <typename T>
//    std::vector<T, AlignedAllocator<T, alignment>> aligned_vector(size_t reserve = 0){
//        std::vector<T, AlignedAllocator<T, alignment>> vec;
//        if(reserve > 0) vec.reserve(reserve);
//        return vec;
//    }

}

#endif //MICROVECDB_ALIGNED_VECTOR_H
