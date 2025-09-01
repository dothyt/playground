#include <stdio.h>

enum Order {
    FIRST,
    SECOND,
    THIRD
};

enum Colors {
    RED,
    GREEN = RED + SECOND,
    BLUE
};

int main() {
    enum Colors color1 = RED;
    enum Colors color2 = GREEN;
    enum Colors color3 = BLUE;

    printf("Color 1: %d\n", color1);
    printf("Color 2: %d\n", color2);
    printf("Color 3: %d\n", color3);

    return 0;
}
