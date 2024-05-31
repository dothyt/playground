#include <utility>
#include <functional>
#include <iostream>

typedef uint64_t (*FuncPtr)(...);

FuncPtr func_ptr[3];

template <class Functor, class... Args>
uint64_t run(Functor&& f, Args&&... args) {
    return f(std::forward<Args>(args)...);
}

void f1(int a, int b) {
    std::cout << "f1: " << a << " " << b << std::endl;
}

int f2(int a, int b, int c) {
    std::cout << "f2: " << a << " " << b << " " << c << std::endl;
    return a + b + c;
}

int main() {
    func_ptr[0] = (FuncPtr)f1;
    func_ptr[1] = (FuncPtr)f2;
    uint64_t y1 = run(func_ptr[0], 1, 2);
    uint64_t y2 = run(func_ptr[1], 3, 4, 5);
    std::cout << "y1: " << y1 << " y2: " << y2 << std::endl;
    return 0;
}