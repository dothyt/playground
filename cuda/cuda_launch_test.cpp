#include <stdio.h>
#include <stdlib.h>

#define REPEATS       (2)
#define LAUNCHES      (1000000)
#define USEC_PER_SEC  (1000000)

// Timer with microsecond resolution
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() * 1.0e-3;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}
#else
#error unsupported platform
#endif

__global__ void kernel (void) {}

int main (void)
{
    double start, stop, elapsed;
    int i;

    printf ("\nlaunch overhead WITHOUT per-kernel synchronization\n");
    i = REPEATS;
    do {
        cudaDeviceSynchronize(); // wait until any pending activities complete
        start = second();
        for (int j = 0; j < LAUNCHES; j++) {
            kernel<<<1,1>>>();
        }
        cudaDeviceSynchronize();
        stop = second();
        elapsed = stop - start;
        i--;
        if (!i) printf ("%d launches. Elapsed = %.6f seconds: %f usec/launch\n",
                        LAUNCHES, elapsed, (elapsed / LAUNCHES) * USEC_PER_SEC);
    } while (i);
    printf ("\nlaunch overhead WITH per-kernel synchronization\n");
    i = REPEATS;
    do {
        cudaDeviceSynchronize(); // wait until any pending activities complete
        start = second();
        for (int j = 0; j < LAUNCHES; j++) {
            kernel<<<1,1>>>();
            cudaDeviceSynchronize();
        }
        stop = second();
        elapsed = stop - start;
        i--;
        if (!i) printf ("%d launches. Elapsed = %.6f seconds: %f usec/launch\n",
                        LAUNCHES, elapsed, (elapsed / LAUNCHES) * USEC_PER_SEC);
    } while (i);
    return EXIT_SUCCESS;
}