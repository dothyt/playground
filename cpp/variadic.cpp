#include <iostream>
#include <array>

class C {
    int c;
};

class D {
    int D;
};

template <typename T, typename NOTUSE, size_t N>
class A {
public:
    const std::array<int, N> arr;
    const uint64_t a = 0;
    T t;

    template <typename... Args>
    A(T t, Args... args) 
        : t(t)
        , arr{args...} 
    {}
};

int main() {
    C c;
    A<C, D, 5> a(c, 2, 3, 4, 5, 6);

    for (int i = 0; i < a.arr.size(); i++) {
        std::cout << a.arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}