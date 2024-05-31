#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <stdio.h>

inline float compute_max(__m256 ymm_max_lo) {
    __m128 xmm_max_lo = _mm256_castps256_ps128(ymm_max_lo);
    __m128 xmm_max_hi = _mm256_extractf32x4_ps(ymm_max_lo, 1);
    xmm_max_lo = _mm_max_ps(xmm_max_lo, xmm_max_hi);

    xmm_max_hi = _mm_permute_ps(xmm_max_lo, 0x0E);
    xmm_max_lo = _mm_max_ps(xmm_max_lo, xmm_max_hi);

    xmm_max_hi = _mm_permute_ps(xmm_max_lo, 0x01);
    xmm_max_lo = _mm_max_ps(xmm_max_lo, xmm_max_hi);
    return _mm_cvtss_f32(xmm_max_lo);
}

inline __m512i mm( __m512i a, __m512i b, __m512i c, __m512i z) {
    __m512i d = _mm512_maddubs_epi16(a, b);
    __m512i e = _mm512_maddubs_epi16(b, c);
    __m512i f = _mm512_madd_epi16(d, e);

    __m512i g = _mm512_maddubs_epi16(c, a);
    __m512i h = _mm512_maddubs_epi16(a, z);
    __m512i i = _mm512_madd_epi16(g, h);

    return _mm512_add_epi32(f, i);
}

inline __m256i scale_m256_int8(__m256 x) {
    // use 1/sqrt(x^2) to find max x so that no division needed
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x2_rsqrt = _mm256_rsqrt_ps(x2);
    __m128 xmm_min_lo = _mm256_castps256_ps128(x2_rsqrt);
    __m128 xmm_min_hi = _mm256_extractf32x4_ps(x2_rsqrt, 1);
    xmm_min_lo = _mm_min_ps(xmm_min_lo, xmm_min_hi);

    xmm_min_hi = _mm_permute_ps(xmm_min_lo, 0x0E);
    xmm_min_lo = _mm_min_ps(xmm_min_lo, xmm_min_hi);

    xmm_min_hi = _mm_permute_ps(xmm_min_lo, 0x01);
    xmm_min_lo = _mm_min_ps(xmm_min_lo, xmm_min_hi);

    const __m256 _multiplier_255 = _mm256_set1_ps(63.0f);
    __m256 min_broadcast = _mm256_broadcast_f32x2(xmm_min_lo);
    min_broadcast = _mm256_permute_ps(min_broadcast, 0);    // dup min value 4 times
    __m256 scale = _mm256_mul_ps(min_broadcast, _multiplier_255);
    __m256 x_scaled = _mm256_mul_ps(x, scale);
    __m256i x_scaled_int = _mm256_cvtps_epi32(x_scaled);
    // 1,2,3,4,5,6,7,8 -> (1,2,3,4)x2, (5,6,7,8)x2
    __m256i x_scaled_short = _mm256_packus_epi32(x_scaled_int, x_scaled_int);
    // (1,2,3,4)x2, (5,6,7,8)x2 -> (1,2,3,4)x4, (5,6,7,8)x4
    __m256i x_scaled_int8 = _mm256_packus_epi16(x_scaled_short, x_scaled_short);
    return x_scaled_int8;
}

void print_avx256(__m256i a) {
    uint8_t memory[256] = {0};
    _mm256_storeu_si256((__m256i *)memory, a);
    for (int i = 0; i < 32; i++)
        std::cout << std::hex << (int)memory[i] << " ";
    std::cout << std::dec << std::endl;
}

int main() {
    __m256 ymm_max_lo = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    __m512i a = _mm512_set1_epi8(1);
    __m512i b = _mm512_set1_epi8(2);
    __m512i c = _mm512_set1_epi8(3);
    __m512i z = _mm512_set1_epi8(4);

    float result = 0;
    const uint64_t iters = 10000000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        result += compute_max(ymm_max_lo);
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    std::cout << "Result: " << result << std::endl;
    std::cout << iters * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / iters << " ns/op" << std::endl;

    __m512i result2 = _mm512_set1_epi32(0);
    __m512i result3 = _mm512_set1_epi32(0);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        result2 = mm(a, b, c, z);
        result3 = _mm512_add_epi32(result2, result3);
    }
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    std::cout << "Result: " << result3[0] << std::endl;
    std::cout << iters * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / iters << " ns/op" << std::endl;

    __m256 x = _mm256_setr_ps(1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f);
    __m256i y = _mm256_set1_epi8(0);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        y = scale_m256_int8(x);
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    print_avx256(y);
    std::cout << iters * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / iters << " ns/op" << std::endl;

    __m512i test = _mm512_setzero_si512();
    __m512i target = _mm512_setzero_si512();
    uint64_t addr = 0x123456789abcdef0;
    uint64_t x2[128] __attribute__((aligned(64))) = {0};
    __mmask8 mask = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        addr++;
        target = _mm512_set1_epi64(addr);
        for (int j = 0; j < 16; j++) {
            test = _mm512_load_si512((__m512i *)&x2[j << 3]);
            mask += _mm512_cmpeq_epi64_mask(test, target);
        }

    }
        
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    print_avx256(y);
    std::cout << iters * 1000000000 / time << " ops/s" << std::endl;
    std::cout << time / iters << " ns/op" << std::endl;
    std::cout << "Mask: " << (int)mask << std::endl;

    return 0;
}

