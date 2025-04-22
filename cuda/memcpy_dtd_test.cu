#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector> // Using vector for easier stream management

// --- Constants ---
// Size of each memory copy in bytes
#define COPY_SIZE 1024
// Number of copies to perform for timing
#define NUM_COPIES 1000000 // Total number of copies
// Number of streams to create in the pool
#define NUM_STREAMS 1000  // Fixed number of streams for round-robin assignment

// --- CUDA Error Checking Macro ---
// Macro to wrap CUDA calls and check for errors
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        /* Cleanup resources before exiting */ \
        /* Note: This cleanup is basic; a real application might need more robust handling */ \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- Main Function ---
int main() {
    // --- Device Information (Optional) ---
    int deviceId;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDevice(&deviceId));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, deviceId));
    printf("Using CUDA Device %d: %s\n", deviceId, deviceProp.name);
    printf("Performing %d asynchronous D2D copies of %d bytes each.\n", NUM_COPIES, COPY_SIZE);
    printf("Using a pool of %d streams in a round-robin fashion.\n", NUM_STREAMS);
    printf("--------------------------------------------------\n");

    // --- Variable Declarations ---
    void *d_src = NULL;         // Pointer to source memory on the device
    void *d_dst = NULL;         // Pointer to destination memory on the device
    std::vector<cudaStream_t> streams(NUM_STREAMS); // Vector to hold the stream pool
    cudaEvent_t start, stop;    // CUDA events for timing
    float elapsed_time_ms = 0.0f; // Elapsed time in milliseconds
    double total_data_gb = 0.0;   // Total data transferred in Gigabytes
    double throughput_gbps = 0.0; // Calculated throughput in GB/s

    // --- Resource Allocation ---
    // Allocate memory on the default device
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, COPY_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dst, COPY_SIZE));
    printf("Allocated %d bytes for source buffer on device.\n", COPY_SIZE);
    printf("Allocated %d bytes for destination buffer on device.\n", COPY_SIZE);

    // Create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    printf("Created CUDA events for timing.\n");

    // Create the pool of CUDA streams
    printf("Creating %d CUDA streams for the pool...\n", NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    printf("Finished creating stream pool.\n");


    // --- Warm-up (Optional but recommended) ---
    // Perform a single copy on stream 0 for warm-up
    printf("Performing warm-up copy...\n");
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_dst, d_src, COPY_SIZE, cudaMemcpyDeviceToDevice, streams[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[0])); // Wait for warm-up copy
    printf("Warm-up complete.\n");

    // --- Timed Copy Operations ---
    printf("Starting timed copy loop (round-robin assignment to %d streams)...\n", NUM_STREAMS);
    // Record the start event on the default stream (stream 0) before launching tasks
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0)); // Record start time

    // Launch the asynchronous copies, assigning streams round-robin
    for (int i = 0; i < NUM_COPIES; ++i) {
        // Select the stream using modulo operator
        cudaStream_t current_stream = streams[i % NUM_STREAMS];
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_dst, d_src, COPY_SIZE, cudaMemcpyDeviceToDevice, current_stream));
    }

    // Record the stop event on the default stream (stream 0) after launching all copies
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0)); // Record stop time *after* launching all copies

    // --- Synchronization and Timing ---
    printf("Synchronizing all streams via event...\n");
    // Wait for the stop event to complete. This ensures all operations enqueued
    // on *any* stream before the event record call are finished.
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    printf("Finished timed copy loop and synchronization.\n");


    // Calculate the elapsed time between the start and stop events
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

    // --- Throughput Calculation ---
    // Total data transferred = number of copies * size of each copy
    total_data_gb = (double)NUM_COPIES * COPY_SIZE / (1024.0 * 1024.0 * 1024.0); // Convert bytes to GB

    // Throughput = Total Data (GB) / Elapsed Time (seconds)
    throughput_gbps = total_data_gb / (elapsed_time_ms / 1000.0); // Convert ms to s

    // --- Results ---
    printf("--------------------------------------------------\n");
    printf("Results (Round-Robin Streams):\n");
    printf("Elapsed Time: %.4f ms\n", elapsed_time_ms);
    printf("Total Data Transferred: %.6f GB\n", total_data_gb);
    printf("Effective Throughput (cudaMemcpyAsync D2D): %.4f GB/s\n", throughput_gbps);
    printf("--------------------------------------------------\n");

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    if (d_src) CHECK_CUDA_ERROR(cudaFree(d_src));
    if (d_dst) CHECK_CUDA_ERROR(cudaFree(d_dst));

    // Destroy all created streams in the pool
    printf("Destroying %d CUDA streams...\n", NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        if (streams[i]) CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
    printf("Finished destroying streams.\n");

    // Destroy events
    if (start) CHECK_CUDA_ERROR(cudaEventDestroy(start));
    if (stop) CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("Cleanup complete.\n");

    // Reset device to ensure clean state for subsequent CUDA interactions (optional)
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
