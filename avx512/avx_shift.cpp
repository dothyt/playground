#include <immintrin.h>

template<int N>
__m512i shift_right(__m512i a, __m512i carry = _mm512_setzero_si512())
{
  static_assert(0 <= N && N <= 64);
  if constexpr(N   == 0) return a;
  if constexpr(N   ==64) return carry;
  if constexpr(N%4 == 0) return _mm512_alignr_epi32(carry, a, N / 4);
  else
  {
    __m512i a0 = shift_right< (N/16 + 1)*16>(a, carry);  // 16, 32, 48, 64
    __m512i a1 = shift_right< (N/16    )*16>(a, carry);  //  0, 16, 32, 48
    return _mm512_alignr_epi8(a0, a1, N % 16);
  }
}

template<int N>
__m512i shift_left(__m512i a, __m512i carry = _mm512_setzero_si512())
{
  return shift_right<64-N>(carry, a);
}

// Examples to see generated assembly:
__m512i shiftleft_1(__m512i a)
{
    return shift_left<1>(a);
}

#include <iostream>
#include <iomanip>

void print(__m512i a)
{
    alignas(64) unsigned char temp[64];
    _mm512_store_si512(temp, a);
    for(int i=64; 
        i --> 0; // down-to operator ;)
        std::cout << std::hex << std::setw(2) <<std::setfill('0') << int(temp[i])
    );
}


#include <chrono>
int main()
{
    __m512i x = _mm512_set_epi64(
        0x3f3e3d3c3b3a3938,0x3736353433323130,
        0x2f2e2d2c2b2a2928,0x2726252423222120,
        0x1f1e1d1c1b1a1918,0x1716151413121110,
        0x0f0e0d0c0b0a0908,0x0706050403020100
        );
    __m512i y = _mm512_set_epi64(
        0x7f7e7d7c7b7a7978,0x7776757473727170,
        0x6f6e6d6c6b6a6968,0x6766656463626160,
        0x5f5e5d5c5b5a5958,0x5756555453525150,
        0x4f4e4d4c4b4a4948,0x4746454443424140
        );
    
    uint64_t iters = 100000000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        shift_right<1>(x,y);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "shift_left<1> time: " << time << "ns\n";
    std::cout << "shift_left<1> time : " << (float)time / iters << "ns\n";

    // print(shift_left<1>(x,y)); std::cout << '\n';
    // print(shift_right<0x00>(x,y)); std::cout << '\n';
    // print(shift_right<0x01>(x,y)); std::cout << '\n';
    // print(shift_right<0x02>(x,y)); std::cout << '\n';
    // print(shift_right<0x03>(x,y)); std::cout << '\n';
    // print(shift_right<0x04>(x,y)); std::cout << '\n';
    // print(shift_right<0x05>(x,y)); std::cout << '\n';
    // print(shift_right<0x06>(x,y)); std::cout << '\n';
    // print(shift_right<0x07>(x,y)); std::cout << '\n';
    // print(shift_right<0x08>(x,y)); std::cout << '\n';
    // print(shift_right<0x09>(x,y)); std::cout << '\n';
    // print(shift_right<0x0a>(x,y)); std::cout << '\n';
    // print(shift_right<0x0b>(x,y)); std::cout << '\n';
    // print(shift_right<0x0c>(x,y)); std::cout << '\n';
    // print(shift_right<0x0d>(x,y)); std::cout << '\n';
    // print(shift_right<0x0e>(x,y)); std::cout << '\n';
    // print(shift_right<0x0f>(x,y)); std::cout << '\n';
    // print(shift_right<0x10>(x,y)); std::cout << '\n';
    // print(shift_right<0x11>(x,y)); std::cout << '\n';
    // print(shift_right<0x12>(x,y)); std::cout << '\n';
    // print(shift_right<0x13>(x,y)); std::cout << '\n';
    // print(shift_right<0x14>(x,y)); std::cout << '\n';
    // print(shift_right<0x15>(x,y)); std::cout << '\n';
    // print(shift_right<0x16>(x,y)); std::cout << '\n';
    // print(shift_right<0x17>(x,y)); std::cout << '\n';
    // print(shift_right<0x18>(x,y)); std::cout << '\n';
    // print(shift_right<0x19>(x,y)); std::cout << '\n';
    // print(shift_right<0x1a>(x,y)); std::cout << '\n';
    // print(shift_right<0x1b>(x,y)); std::cout << '\n';
    // print(shift_right<0x1c>(x,y)); std::cout << '\n';
    // print(shift_right<0x1d>(x,y)); std::cout << '\n';
    // print(shift_right<0x1e>(x,y)); std::cout << '\n';
    // print(shift_right<0x1f>(x,y)); std::cout << '\n';
    // print(shift_right<0x20>(x,y)); std::cout << '\n';
    // print(shift_right<0x21>(x,y)); std::cout << '\n';
    // print(shift_right<0x22>(x,y)); std::cout << '\n';
    // print(shift_right<0x23>(x,y)); std::cout << '\n';
    // print(shift_right<0x24>(x,y)); std::cout << '\n';
    // print(shift_right<0x25>(x,y)); std::cout << '\n';
    // print(shift_right<0x26>(x,y)); std::cout << '\n';
    // print(shift_right<0x27>(x,y)); std::cout << '\n';
    // print(shift_right<0x28>(x,y)); std::cout << '\n';
    // print(shift_right<0x29>(x,y)); std::cout << '\n';
    // print(shift_right<0x2a>(x,y)); std::cout << '\n';
    // print(shift_right<0x2b>(x,y)); std::cout << '\n';
    // print(shift_right<0x2c>(x,y)); std::cout << '\n';
    // print(shift_right<0x2d>(x,y)); std::cout << '\n';
    // print(shift_right<0x2e>(x,y)); std::cout << '\n';
    // print(shift_right<0x2f>(x,y)); std::cout << '\n';
    // print(shift_right<0x30>(x,y)); std::cout << '\n';
    // print(shift_right<0x31>(x,y)); std::cout << '\n';
    // print(shift_right<0x32>(x,y)); std::cout << '\n';
    // print(shift_right<0x33>(x,y)); std::cout << '\n';
    // print(shift_right<0x34>(x,y)); std::cout << '\n';
    // print(shift_right<0x35>(x,y)); std::cout << '\n';
    // print(shift_right<0x36>(x,y)); std::cout << '\n';
    // print(shift_right<0x37>(x,y)); std::cout << '\n';
    // print(shift_right<0x38>(x,y)); std::cout << '\n';
    // print(shift_right<0x39>(x,y)); std::cout << '\n';
    // print(shift_right<0x3a>(x,y)); std::cout << '\n';
    // print(shift_right<0x3b>(x,y)); std::cout << '\n';
    // print(shift_right<0x3c>(x,y)); std::cout << '\n';
    // print(shift_right<0x3d>(x,y)); std::cout << '\n';
    // print(shift_right<0x3e>(x,y)); std::cout << '\n';
    // print(shift_right<0x3f>(x,y)); std::cout << '\n';
    // print(shift_right<0x40>(x,y)); std::cout << '\n';

}

