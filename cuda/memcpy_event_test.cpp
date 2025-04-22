#include <iostream>
#include <vector>
#include <numeric> // Required for std::iota
#include <stdexcept> // Required for std::runtime_error
#include <cuda_runtime.h>

/**
 * @brief CUDA error checking macro.
 * Checks the result of a CUDA Runtime API call and exits if an error occurred.
 * @param ans The CUDA API call result.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code); // Exit program if a CUDA error occurs
   }
}

/**
 * @brief Verifies if the contents of two vectors are identical.
 * @param expected The vector containing the expected values.
 * @param actual The vector containing the actual values received.
 * @param name A descriptive name for the verification step.
 * @return True if verification passes, false otherwise.
 */
bool verify(const std::vector<int>& expected, const std::vector<int>& actual, const std::string& name) {
    if (expected.size() != actual.size()) {
        std::cerr << "Verification failed for " << name << ": Size mismatch! Expected "
                  << expected.size() << ", Got " << actual.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            // Print only the first mismatch found for brevity
            std::cerr << "Verification failed for " << name << " at index " << i
                      << ". Expected: " << expected[i] << ", Got: " << actual[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification successful for " << name << "." << std::endl;
    return true;
}


int main() {
    // --- 1. Initialization and Device Check ---
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Error: This example requires at least 2 CUDA-enabled GPUs." << std::endl;
        std::cerr << "Found " << deviceCount << " devices. Exiting." << std::endl;
        return 1; // Indicate failure
    }
    std::cout << "Found " << deviceCount << " CUDA devices. Using GPU 0 and GPU 1." << std::endl;
    const int dev0 = 0;
    const int dev1 = 1;

    // --- 2. Enable Peer Access ---
    int canAccessPeer01, canAccessPeer10;
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1));
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0));

    if (!canAccessPeer01 || !canAccessPeer10) {
         std::cerr << "Error: Peer access is not supported between GPU " << dev0 << " and GPU " << dev1 << "." << std::endl;
         if (!canAccessPeer01) std::cerr << " -> GPU " << dev0 << " cannot directly access memory on GPU " << dev1 << std::endl;
         if (!canAccessPeer10) std::cerr << " -> GPU " << dev1 << " cannot directly access memory on GPU " << dev0 << std::endl;
         return 1; // Indicate failure
    }
    std::cout << "Peer access is supported between GPU " << dev0 << " and GPU " << dev1 << "." << std::endl;

    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaDeviceEnablePeerAccess(dev1, 0));
    std::cout << "Enabled peer access for GPU " << dev0 << " to access GPU " << dev1 << "'s memory." << std::endl;

    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaDeviceEnablePeerAccess(dev0, 0));
    std::cout << "Enabled peer access for GPU " << dev1 << " to access GPU " << dev0 << "'s memory." << std::endl;


    // --- 3. Define Buffer Size and Allocate Host Memory ---
    const size_t N = 1024 * 1024 * 32; // 32 Million integers
    const size_t bufferSize = N * sizeof(int);
    std::cout << "Buffer size per allocation: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;

    std::vector<int> h_src0_A(N), h_src0_B(N);
    std::vector<int> h_src1_A(N), h_src1_B(N);
    std::vector<int> h_verify1_A(N), h_verify1_B(N);
    std::vector<int> h_verify0_A(N), h_verify0_B(N);

    std::cout << "Initializing host data..." << std::endl;
    std::iota(h_src0_A.begin(), h_src0_A.end(), 0);        // Pattern 0
    std::iota(h_src0_B.begin(), h_src0_B.end(), N);        // Pattern 1
    std::iota(h_src1_A.begin(), h_src1_A.end(), 2 * N);    // Pattern 2
    std::iota(h_src1_B.begin(), h_src1_B.end(), 3 * N);    // Pattern 3

    // --- 4. Allocate Device Memory ---
    int *d_src0_A = nullptr, *d_src0_B = nullptr, *d_dst0_A = nullptr, *d_dst0_B = nullptr;
    int *d_src1_A = nullptr, *d_src1_B = nullptr, *d_dst1_A = nullptr, *d_dst1_B = nullptr;

    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaMalloc(&d_src0_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src0_B, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst0_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst0_B, bufferSize));
    std::cout << "Allocated source and destination buffers on GPU " << dev0 << "." << std::endl;

    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaMalloc(&d_src1_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src1_B, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst1_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst1_B, bufferSize));
    std::cout << "Allocated source and destination buffers on GPU " << dev1 << "." << std::endl;

    // --- 5. Create CUDA Streams and Event ---
    cudaStream_t stream1, stream2;
    cudaEvent_t stream1DoneEvent; // <<< Declare the event

    gpuErrchk(cudaSetDevice(dev0)); // Create streams/event on dev0 context
    gpuErrchk(cudaStreamCreate(&stream1));
    gpuErrchk(cudaEventCreate(&stream1DoneEvent)); // <<< Create the event
    // trying this out
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaStreamCreate(&stream2));
    
    std::cout << "Created two CUDA streams and one CUDA event." << std::endl;

    // --- 6. Copy Initial Data from Host to Device Source Buffers ---
    std::cout << "Copying initial data from Host to Device source buffers..." << std::endl;
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaMemcpy(d_src0_A, h_src0_A.data(), bufferSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src0_B, h_src0_B.data(), bufferSize, cudaMemcpyHostToDevice));

    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaMemcpy(d_src1_A, h_src1_A.data(), bufferSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src1_B, h_src1_B.data(), bufferSize, cudaMemcpyHostToDevice));

    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaDeviceSynchronize());
    std::cout << "Initial data is ready on both GPUs." << std::endl;


    // --- 7. Perform Asynchronous Peer-to-Peer Copies with Event Dependency ---
    std::cout << "Starting asynchronous peer-to-peer copies with dependency..." << std::endl;
    std::cout << "(Stream2 will wait for Stream1 to complete its tasks)" << std::endl;

    // --- Enqueue operations for Stream 1 ---
    // Set context for initiating device (optional but good practice)
    gpuErrchk(cudaSetDevice(dev0));
    // Transfer 1 (S1): GPU 0 Source A -> GPU 1 Destination A
    std::cout << "  Enqueueing on Stream1: GPU0::d_src0_A -> GPU1::d_dst1_A" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_A, dev1, d_src0_A, dev0, bufferSize, stream1));

    // Set context for initiating device (optional but good practice)
    gpuErrchk(cudaSetDevice(dev1));
    // Transfer 2 (S1): GPU 1 Source A -> GPU 0 Destination A
    std::cout << "  Enqueueing on Stream1: GPU1::d_src1_A -> GPU0::d_dst0_A" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst0_A, dev0, d_src1_A, dev1, bufferSize, stream1));

    // --- Record event on Stream 1 ---
    // This marks the point in stream1 that stream2 needs to wait for.
    // The event is considered "complete" only when ALL preceding work in stream1 finishes.
    std::cout << "  Recording event on Stream1 (stream1DoneEvent)" << std::endl;
    gpuErrchk(cudaEventRecord(stream1DoneEvent, stream1)); // <<< Record event

    // --- Make Stream 2 Wait for Stream 1 ---
    // This command is enqueued onto stream2. Operations submitted later to stream2
    // will not begin execution until stream1DoneEvent has been recorded AND completed.
    std::cout << "  Enqueueing wait command on Stream2 for stream1DoneEvent" << std::endl;
    gpuErrchk(cudaStreamWaitEvent(stream2, stream1DoneEvent, 0)); // <<< Make stream2 wait (flag must be 0)

    // --- Enqueue operations for Stream 2 (will execute only after stream1's work is done) ---
    // Set context for initiating device
    gpuErrchk(cudaSetDevice(dev0));
    // Transfer 3 (S2): GPU 0 Source B -> GPU 1 Destination B
    std::cout << "  Enqueueing on Stream2: GPU0::d_src0_B -> GPU1::d_dst1_B" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_B, dev1, d_src0_B, dev0, bufferSize, stream2));

    // Set context for initiating device
    gpuErrchk(cudaSetDevice(dev1));
    // Transfer 4 (S2): GPU 1 Source B -> GPU 0 Destination B
    std::cout << "  Enqueueing on Stream2: GPU1::d_src1_B -> GPU0::d_dst0_B" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst0_B, dev0, d_src1_B, dev1, bufferSize, stream2));


    std::cout << "All peer-to-peer copy commands have been issued with dependency." << std::endl;

    // --- 8. Synchronize Streams ---
    // Wait for all operations on BOTH streams to complete. Since stream2 depends
    // on stream1, synchronizing stream2 implicitly waits for stream1 as well.
    // However, explicitly synchronizing both is clear and safe.
    std::cout << "Synchronizing streams to wait for all copies to complete..." << std::endl;
    gpuErrchk(cudaSetDevice(dev0)); // Set context to where streams were created
    gpuErrchk(cudaStreamSynchronize(stream1)); // Wait for stream1's tasks
    gpuErrchk(cudaStreamSynchronize(stream2)); // Wait for stream2's tasks (which includes waiting for stream1)
    std::cout << "Streams synchronized. All P2P copies should be complete." << std::endl;


    // --- 9. Copy Results Back to Host for Verification ---
    std::cout << "Copying results from Device destination buffers back to Host..." << std::endl;
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaMemcpy(h_verify1_A.data(), d_dst1_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify1_B.data(), d_dst1_B, bufferSize, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaMemcpy(h_verify0_A.data(), d_dst0_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify0_B.data(), d_dst0_B, bufferSize, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaDeviceSynchronize());


    // --- 10. Verification ---
    std::cout << "Verifying copied data..." << std::endl;
    bool success = true;
    success &= verify(h_src0_A, h_verify1_A, "GPU0[SrcA] -> GPU1[DstA] (on stream1)");
    success &= verify(h_src1_A, h_verify0_A, "GPU1[SrcA] -> GPU0[DstA] (on stream1)");
    success &= verify(h_src0_B, h_verify1_B, "GPU0[SrcB] -> GPU1[DstB] (on stream2 after wait)");
    success &= verify(h_src1_B, h_verify0_B, "GPU1[SrcB] -> GPU0[DstB] (on stream2 after wait)");

    // --- 11. Cleanup ---
    std::cout << "Cleaning up resources..." << std::endl;
    // Destroy event and streams (use context where they were created)
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaEventDestroy(stream1DoneEvent)); // <<< Destroy the event
    gpuErrchk(cudaStreamDestroy(stream1));
    gpuErrchk(cudaStreamDestroy(stream2));

    // Free device memory
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaFree(d_src0_A));
    gpuErrchk(cudaFree(d_src0_B));
    gpuErrchk(cudaFree(d_dst0_A));
    gpuErrchk(cudaFree(d_dst0_B));

    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaFree(d_src1_A));
    gpuErrchk(cudaFree(d_src1_B));
    gpuErrchk(cudaFree(d_dst1_A));
    gpuErrchk(cudaFree(d_dst1_B));

    std::cout << "\n--- Execution Summary ---" << std::endl;
    if (success) {
        std::cout << "All peer-to-peer copies completed and verified successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cerr << "One or more peer-to-peer copies failed verification." << std::endl;
        return 1; // Indicate failure
    }
}