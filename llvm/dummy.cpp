

void func() {
    // A
    foreach (int i, 0, 10) {
        // B
        if (cond) {
            // C
        } else {
            // D
        }
        // E
    }
    // F
}

int table [] = {
    // A, B, C, E, B, D, E, B, C, E, B, D, E, ...
    // B, C, E, B, C, E, B, C, E, B, C, E, B, C, E, ...
    // B, D, E, B, D, E, B, D, E, B, D, E, B, D, E, ...
    // ... 
};

