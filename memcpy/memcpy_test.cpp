#include <chrono>
#include <iostream>
#include <algorithm>
#include <string.h>
#include <vector>
#include <immintrin.h>

#define ARRAY_SIZE 1000000UL

struct page {
    char data[4096];
} __attribute__((aligned(64)));

class Stopwatch2 {
  uint64_t begin_;
  /// A list of recorded intervals.
  std::vector<uint64_t> intervals_;

public:
  Stopwatch2() : begin_(0) {}

  void start() { begin_ = __builtin_ia32_rdtsc(); }

  void stop() {
    uint64_t interval = __builtin_ia32_rdtsc() - begin_;
    intervals_.push_back(interval);
  }

  uint64_t get_median() {
    std::sort(intervals_.begin(), intervals_.end());
    return intervals_[intervals_.size() / 2];
  }
  uint64_t get_ninety() {
    std::sort(intervals_.begin(), intervals_.end());
    return intervals_[intervals_.size() / 10 * 9];
  }
  uint64_t get_max() {
    std::sort(intervals_.begin(), intervals_.end());
    return intervals_[intervals_.size() - 2];
  }
};

class Stopwatch {
  using time_point = std::chrono::high_resolution_clock::time_point;
  /// The time of the last sample;
  time_point begin_;
  /// A list of recorded intervals.
  std::vector<uint64_t> intervals_;

public:
  Stopwatch() : begin_() {}

  void start() { begin_ = std::chrono::high_resolution_clock::now(); }

  void stop() {
    time_point end = std::chrono::high_resolution_clock::now();
    uint64_t interval =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin_)
            .count();
    intervals_.push_back(interval);
  }

  uint64_t get_median() {
    std::sort(intervals_.begin(), intervals_.end());
    return intervals_[intervals_.size() / 2];
  }
  uint64_t get_ninety() {
    std::sort(intervals_.begin(), intervals_.end());
    return intervals_[intervals_.size() / 10 * 9];
  }
  uint64_t get_max() {
    std::sort(intervals_.begin(), intervals_.end());
    return intervals_[intervals_.size() - 2];
  }
};

#if 0
void avx512_memcpy(void *dst, void *src, size_t size) {
    size_t i = 0;
    for (; i + 512 < size; i += 512) {
        __m512i a1 = _mm512_load_si512((__m512i *)((char *)src + i));
        __m512i a2 = _mm512_load_si512((__m512i *)((char *)src + i + 0x40));
        __m512i a3 = _mm512_load_si512((__m512i *)((char *)src + i + 0x80));
        __m512i a4 = _mm512_load_si512((__m512i *)((char *)src + i + 0xc0));
        __m512i a5 = _mm512_load_si512((__m512i *)((char *)src + i + 0x100));
        __m512i a6 = _mm512_load_si512((__m512i *)((char *)src + i + 0x140));
        __m512i a7 = _mm512_load_si512((__m512i *)((char *)src + i + 0x180));
        __m512i a8 = _mm512_load_si512((__m512i *)((char *)src + i + 0x1c0));

        _mm_prefetch((char *)dst + i, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x40, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x80, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0xc0, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x100, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x140, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x180, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x1c0, _MM_HINT_T1);

        _mm512_store_si512((__m512i *)((char *)dst + i), a1);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x40), a2);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x80), a3);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0xc0), a4);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x100), a5);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x140), a6);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x180), a7);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x1c0), a8);

    }
}

static inline void avx512_memcpy_1024(void *dst, void *src, size_t size) {
    for (size_t i = 0; i < size; i += 1024) {
        __m512i a1 = _mm512_load_si512((__m512i *)((char *)src + i));
        __m512i a2 = _mm512_load_si512((__m512i *)((char *)src + i + 0x40));
        __m512i a3 = _mm512_load_si512((__m512i *)((char *)src + i + 0x80));
        __m512i a4 = _mm512_load_si512((__m512i *)((char *)src + i + 0xc0));
        __m512i a5 = _mm512_load_si512((__m512i *)((char *)src + i + 0x100));
        __m512i a6 = _mm512_load_si512((__m512i *)((char *)src + i + 0x140));
        __m512i a7 = _mm512_load_si512((__m512i *)((char *)src + i + 0x180));
        __m512i a8 = _mm512_load_si512((__m512i *)((char *)src + i + 0x1c0));
        __m512i a9 = _mm512_load_si512((__m512i *)((char *)src + i + 0x200));
        __m512i a10 = _mm512_load_si512((__m512i *)((char *)src + i + 0x240));
        __m512i a11 = _mm512_load_si512((__m512i *)((char *)src + i + 0x280));
        __m512i a12 = _mm512_load_si512((__m512i *)((char *)src + i + 0x2c0));
        __m512i a13 = _mm512_load_si512((__m512i *)((char *)src + i + 0x300));
        __m512i a14 = _mm512_load_si512((__m512i *)((char *)src + i + 0x340));
        __m512i a15 = _mm512_load_si512((__m512i *)((char *)src + i + 0x380));
        __m512i a16 = _mm512_load_si512((__m512i *)((char *)src + i + 0x3c0));

        _mm_prefetch((char *)dst + i, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x40, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x80, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0xc0, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x100, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x140, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x180, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x1c0, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x200, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x240, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x280, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x2c0, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x300, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x340, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x380, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x3c0, _MM_HINT_T1);

        _mm512_store_si512((__m512i *)((char *)dst + i), a1);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x40), a2);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x80), a3);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0xc0), a4);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x100), a5);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x140), a6);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x180), a7);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x1c0), a8);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x200), a9);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x240), a10);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x280), a11);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x2c0), a12);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x300), a13);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x340), a14);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x380), a15);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x3c0), a16);
    }
}
#endif

static inline void avx512_memcpy(void *dst, void* mid, void *src, size_t size) {
    for (size_t i = 0; i < size; i += 0x200) {
        _mm_prefetch((char *)src + i, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0x40, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0x80, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0xc0, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0x100, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0x140, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0x180, _MM_HINT_T1);
        _mm_prefetch((char *)src + i + 0x1c0, _MM_HINT_T1);

        __m512i a1 = _mm512_load_si512((__m512i *)((char *)mid + i));
        __m512i a2 = _mm512_load_si512((__m512i *)((char *)mid + i + 0x40));
        __m512i a3 = _mm512_load_si512((__m512i *)((char *)mid + i + 0x80));
        __m512i a4 = _mm512_load_si512((__m512i *)((char *)mid + i + 0xc0));
        __m512i a5 = _mm512_load_si512((__m512i *)((char *)mid + i + 0x100));
        __m512i a6 = _mm512_load_si512((__m512i *)((char *)mid + i + 0x140));
        __m512i a7 = _mm512_load_si512((__m512i *)((char *)mid + i + 0x180));
        __m512i a8 = _mm512_load_si512((__m512i *)((char *)mid + i + 0x1c0));

        _mm_prefetch((char *)dst + i, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x40, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x80, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0xc0, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x100, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x140, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x180, _MM_HINT_T1);
        _mm_prefetch((char *)dst + i + 0x1c0, _MM_HINT_T1);
  
        __m512i b1 = _mm512_load_si512((__m512i *)((char *)src + i));
        __m512i b2 = _mm512_load_si512((__m512i *)((char *)src + i + 0x40));
        __m512i b3 = _mm512_load_si512((__m512i *)((char *)src + i + 0x80));
        __m512i b4 = _mm512_load_si512((__m512i *)((char *)src + i + 0xc0));
        __m512i b5 = _mm512_load_si512((__m512i *)((char *)src + i + 0x100));
        __m512i b6 = _mm512_load_si512((__m512i *)((char *)src + i + 0x140));
        __m512i b7 = _mm512_load_si512((__m512i *)((char *)src + i + 0x180));
        __m512i b8 = _mm512_load_si512((__m512i *)((char *)src + i + 0x1c0));

        _mm512_store_si512((__m512i *)((char *)dst + i), a1);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x40), a2);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x80), a3);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0xc0), a4);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x100), a5);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x140), a6);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x180), a7);
        _mm512_store_si512((__m512i *)((char *)dst + i + 0x1c0), a8);

        _mm_prefetch((char *)mid + i + 0x200, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x240, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x280, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x2c0, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x300, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x340, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x380, _MM_HINT_T1);
        _mm_prefetch((char *)mid + i + 0x3c0, _MM_HINT_T1);

        _mm512_store_si512((__m512i *)((char *)mid + i), b1);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0x40), b2);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0x80), b3);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0xc0), b4);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0x100), b5);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0x140), b6);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0x180), b7);
        _mm512_store_si512((__m512i *)((char *)mid + i + 0x1c0), b8);
    }
}

int main() {
    page *src = (page*)aligned_alloc(64, sizeof(page) * ARRAY_SIZE);
    page *dst = (page*)aligned_alloc(64, sizeof(page) * ARRAY_SIZE);
    page *mid = (page*)aligned_alloc(64, sizeof(page) * ARRAY_SIZE);
    const uint64_t iters = 20;
    Stopwatch T;

    for (int j = 0; j < iters; j++) {
        T.start();
        for (int i = 0; i < ARRAY_SIZE; i++) {
            // memcpy(&dst[i], &src[i], sizeof(page));
            // avx512_memcpy_1024(&dst[i], &src[i], sizeof(page));
            avx512_memcpy(&dst[i], &mid[i], &src[i], sizeof(page));
        }
        T.stop();
    }

    auto time = T.get_median();
    std::cout << ARRAY_SIZE * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / ARRAY_SIZE << " ns/op" << std::endl;

    time = T.get_ninety();
    std::cout << ARRAY_SIZE * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / ARRAY_SIZE << " ns/op" << std::endl;

    time = T.get_max();
    std::cout << ARRAY_SIZE * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / ARRAY_SIZE << " ns/op" << std::endl;

    // auto time = T.get_median();
    // std::cout << time / ARRAY_SIZE << " cycle/op" << std::endl;

    // time = T.get_ninety();
    // std::cout << time / ARRAY_SIZE << " cycle/op" << std::endl;

    // time = T.get_max();
    // std::cout << time / ARRAY_SIZE << " cycle/op" << std::endl;
}