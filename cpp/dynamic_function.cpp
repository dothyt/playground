#include <iostream>

template <typename F>
void function (int test) {
    std::cout << "test: " << test << std::endl;
    F();
    std::cout << "function: " << test+1 << std::endl;
}

class MyClass {
public:
    int local;

    MyClass(int c) : local(c) {}

    static void f() {
        std::cout << "MyClass::f()" << std::endl;
    }

    void operator()() {
        std::cout << "MyClass::operator()" << std::endl;
    }
};

void function2() {
    std::cout << "function()" << std::endl;
}

int main() {
    MyClass c(1);
    function<&function2>(1234);
    return 0;
}