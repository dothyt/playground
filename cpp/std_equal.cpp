#include <immintrin.h>
#include <cstdint>
#include <functional>
#include <iostream>

#define PREFETCHER_SEQ_LEN          (64)

#define M512_LEN (PREFETCHER_SEQ_LEN * sizeof(uint16_t) / 64)
union History {
    uint16_t u16[PREFETCHER_SEQ_LEN];
    __m512i m512i[M512_LEN];
};

class CacheEntry {
public:
    History history;
    uint16_t future[PREFETCHER_SEQ_LEN];

    auto operator==(CacheEntry const& other) const -> bool {
        return equal(other);
    }

private:
    inline bool equal(CacheEntry const& other) const {
        uint32_t all_mask = 0;
        __mmask32 mask[M512_LEN];
        std::cout << "this is called\n";

        // early exit
        if (history.u16 != other.history.u16)
            return false;
        for (int i = 0; i < M512_LEN; i++) {
            mask[i] = _mm512_cmpneq_epu16_mask(history.m512i[i], other.history.m512i[i]);
        }
        // TODO: do a weight sum for more tolerance
        for (int i = 0; i < M512_LEN; i++) {
            all_mask += mask[i];
        }
        return all_mask == 0;
    }
};

template <class A, 
          class U = std::equal_to<A> >
bool f(A a, A b, U u = U())
{
    return u(a, b);
}

int main() {
    CacheEntry a, b;
    auto x = f(a, b);
    std::cout << x << std::endl;
    return 0;
}
