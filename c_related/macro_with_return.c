// Type your code here, or load an example.
#include <stdio.h>

#define macro_with_return() \
({ \
    int ret = 3; \
    printf("ret inside macro: %d\n", ret); \
    ret; \
})

int main() {
    int ret = 1;
    printf("ret outside: %d\n", ret);
    ret = macro_with_return();
    printf("ret outside 2: %d\n", ret);
    return 0;
}