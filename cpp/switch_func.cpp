#include <stdint.h>
#include <stdio.h>

extern "C" {
    void func1(int j);
    void func2(int j);
    void func3(int j);
}

uint64_t var1 = 0;
uint64_t var2 = 0;
uint64_t var3 = 0;

void func1(int j) {
    var1 += j;
}

void func2(int j) {
    var2 += j;
}

void func3(int j) {
    var3 += j;
}

int main() {
    for (int i = 0; i < 10; ++i) {
        switch (i) {
            case 0:
                func1(i);
                break;
            case 1:
                func2(i);
                break;
            case 2:
                func3(i);
                break;
            default:
                break;
        }
        printf("var1: %lu, var2: %lu, var3: %lu\n", var1, var2, var3);
    }
    return 0;
}