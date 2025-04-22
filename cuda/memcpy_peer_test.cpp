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
    // Check if peer access is possible between the two devices
    int canAccessPeer01, canAccessPeer10;
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1));
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0));

    if (!canAccessPeer01 || !canAccessPeer10) {
         std::cerr << "Error: Peer access is not supported between GPU " << dev0 << " and GPU " << dev1 << "." << std::endl;
         // Provide more specific feedback
         if (!canAccessPeer01) std::cerr << " -> GPU " << dev0 << " cannot directly access memory on GPU " << dev1 << std::endl;
         if (!canAccessPeer10) std::cerr << " -> GPU " << dev1 << " cannot directly access memory on GPU " << dev0 << std::endl;
         std::cerr << "Peer access is often enabled via NVLink or over PCIe (check system configuration)." << std::endl;
         return 1; // Indicate failure
    }
    std::cout << "Peer access is supported between GPU " << dev0 << " and GPU " << dev1 << "." << std::endl;

    // Enable peer access. This needs to be done for each device that will access the other's memory.
    // Set context to the device that will be *doing the accessing*.
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaDeviceEnablePeerAccess(dev1, 0)); // Flag is reserved, must be 0
    std::cout << "Enabled peer access for GPU " << dev0 << " to access GPU " << dev1 << "'s memory." << std::endl;

    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaDeviceEnablePeerAccess(dev0, 0)); // Flag is reserved, must be 0
    std::cout << "Enabled peer access for GPU " << dev1 << " to access GPU " << dev0 << "'s memory." << std::endl;


    // --- 3. Define Buffer Size and Allocate Host Memory ---
    const size_t N = 1024 * 1024 * 32; // 32 Million integers (adjust as needed)
    const size_t bufferSize = N * sizeof(int); // Size in bytes
    std::cout << "Buffer size per allocation: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;

    // Host buffers for initial data
    std::vector<int> h_src0_A(N), h_src0_B(N); // Data originating from GPU 0
    std::vector<int> h_src1_A(N), h_src1_B(N); // Data originating from GPU 1

    // Host buffers to store results copied back from device for verification
    std::vector<int> h_verify1_A(N), h_verify1_B(N); // To verify copies that landed on GPU 1
    std::vector<int> h_verify0_A(N), h_verify0_B(N); // To verify copies that landed on GPU 0

    // Initialize host source buffers with distinct, predictable data patterns
    std::cout << "Initializing host data..." << std::endl;
    std::iota(h_src0_A.begin(), h_src0_A.end(), 0);        // Pattern 0: 0, 1, 2, ...
    std::iota(h_src0_B.begin(), h_src0_B.end(), N);        // Pattern 1: N, N+1, N+2, ...
    std::iota(h_src1_A.begin(), h_src1_A.end(), 2 * N);    // Pattern 2: 2N, 2N+1, ...
    std::iota(h_src1_B.begin(), h_src1_B.end(), 3 * N);    // Pattern 3: 3N, 3N+1, ...

    // --- 4. Allocate Device Memory ---
    int *d_src0_A = nullptr, *d_src0_B = nullptr, *d_dst0_A = nullptr, *d_dst0_B = nullptr; // Buffers physically on GPU 0
    int *d_src1_A = nullptr, *d_src1_B = nullptr, *d_dst1_A = nullptr, *d_dst1_B = nullptr; // Buffers physically on GPU 1

    // Allocate on GPU 0
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaMalloc(&d_src0_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src0_B, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst0_A, bufferSize)); // Destination for data from GPU 1
    gpuErrchk(cudaMalloc(&d_dst0_B, bufferSize)); // Destination for data from GPU 1
    std::cout << "Allocated source and destination buffers on GPU " << dev0 << "." << std::endl;

    // Allocate on GPU 1
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaMalloc(&d_src1_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src1_B, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst1_A, bufferSize)); // Destination for data from GPU 0
    gpuErrchk(cudaMalloc(&d_dst1_B, bufferSize)); // Destination for data from GPU 0
    std::cout << "Allocated source and destination buffers on GPU " << dev1 << "." << std::endl;

    // --- 5. Create CUDA Streams ---
    cudaStream_t stream1, stream2;
    // Streams are associated with the device active during creation,
    // but can manage operations involving multiple devices if peer access is enabled.
    // It's common practice to create streams on one of the participating devices (e.g., dev0).
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaStreamCreate(&stream1));
    gpuErrchk(cudaStreamCreate(&stream2));
    std::cout << "Created two CUDA streams (stream1 and stream2)." << std::endl;

    // --- 6. Copy Initial Data from Host to Device Source Buffers ---
    // Use the default stream (stream 0) for these initial transfers for simplicity.
    // Ensure these complete before starting the asynchronous peer-to-peer copies.
    std::cout << "Copying initial data from Host to Device source buffers..." << std::endl;
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaMemcpy(d_src0_A, h_src0_A.data(), bufferSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src0_B, h_src0_B.data(), bufferSize, cudaMemcpyHostToDevice));

    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaMemcpy(d_src1_A, h_src1_A.data(), bufferSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src1_B, h_src1_B.data(), bufferSize, cudaMemcpyHostToDevice));

    // Synchronize both devices to guarantee initial data is ready on GPUs before P2P starts.
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaDeviceSynchronize());
    std::cout << "Initial data is ready on both GPUs." << std::endl;


    // --- 7. Perform Asynchronous Peer-to-Peer Copies using Different Streams ---
    std::cout << "Starting asynchronous peer-to-peer copies..." << std::endl;

    // Set context to the device initiating the copy (GPU 0 for the first two)
    // While not strictly required after enabling peer access for the memcpy itself,
    // it's good practice for managing which device's resources are primarily used for launch.
    gpuErrchk(cudaSetDevice(dev0));

    // Transfer 1: GPU 0 Source A -> GPU 1 Destination A (using stream1)
    std::cout << "  Enqueueing: GPU0::d_src0_A -> GPU1::d_dst1_A on stream1" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_A, dev1, d_src0_A, dev0, bufferSize, stream1));

    // Transfer 2: GPU 0 Source B -> GPU 1 Destination B (using stream2)
    std::cout << "  Enqueueing: GPU0::d_src0_B -> GPU1::d_dst1_B on stream2" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_B, dev1, d_src0_B, dev0, bufferSize, stream2));


    // Set context to the device initiating the copy (GPU 1 for the next two)
    gpuErrchk(cudaSetDevice(dev1));

    // Transfer 3: GPU 1 Source A -> GPU 0 Destination A (using stream1)
    std::cout << "  Enqueueing: GPU1::d_src1_A -> GPU0::d_dst0_A on stream1" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst0_A, dev0, d_src1_A, dev1, bufferSize, stream1));

    // Transfer 4: GPU 1 Source B -> GPU 0 Destination B (using stream2)
    std::cout << "  Enqueueing: GPU1::d_src1_B -> GPU0::d_dst0_B on stream2" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst0_B, dev0, d_src1_B, dev1, bufferSize, stream2));

    std::cout << "All peer-to-peer copy commands have been issued asynchronously." << std::endl;
    std::cout << "Copies on stream1 might overlap with copies on stream2 if hardware allows." << std::endl;

    // --- 8. Synchronize Streams ---
    // Wait for all operations enqueued on stream1 and stream2 to complete
    // before proceeding to verification.
    // It's crucial to synchronize the streams on the context they were created with,
    // or use device-wide synchronization.
    std::cout << "Synchronizing streams to wait for copies to complete..." << std::endl;
    gpuErrchk(cudaSetDevice(dev0)); // Set context to where streams were created
    gpuErrchk(cudaStreamSynchronize(stream1));
    gpuErrchk(cudaStreamSynchronize(stream2));
    // Alternatively, a full device synchronize ensures everything is done:
    // gpuErrchk(cudaDeviceSynchronize()); // Syncs all streams on dev0
    // gpuErrchk(cudaSetDevice(dev1));
    // gpuErrchk(cudaDeviceSynchronize()); // Syncs all streams on dev1
    std::cout << "Streams synchronized. All P2P copies should be complete." << std::endl;


    // --- 9. Copy Results Back to Host for Verification ---
    std::cout << "Copying results from Device destination buffers back to Host..." << std::endl;
    // Copy from GPU 1's destination buffers
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaMemcpy(h_verify1_A.data(), d_dst1_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify1_B.data(), d_dst1_B, bufferSize, cudaMemcpyDeviceToHost));

    // Copy from GPU 0's destination buffers
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaMemcpy(h_verify0_A.data(), d_dst0_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify0_B.data(), d_dst0_B, bufferSize, cudaMemcpyDeviceToHost));

    // Ensure DtoH copies are finished (often implicit with blocking memcpy, but good practice)
    gpuErrchk(cudaDeviceSynchronize()); // Sync dev0 after its DtoH
    gpuErrchk(cudaSetDevice(dev1));
    gpuErrchk(cudaDeviceSynchronize()); // Sync dev1 after its DtoH


    // --- 10. Verification ---
    std::cout << "Verifying copied data..." << std::endl;
    bool success = true;
    // Check if data sent from GPU 0 arrived correctly on GPU 1
    success &= verify(h_src0_A, h_verify1_A, "GPU0[SrcA] -> GPU1[DstA] (on stream1)");
    success &= verify(h_src0_B, h_verify1_B, "GPU0[SrcB] -> GPU1[DstB] (on stream2)");

    // Check if data sent from GPU 1 arrived correctly on GPU 0
    success &= verify(h_src1_A, h_verify0_A, "GPU1[SrcA] -> GPU0[DstA] (on stream1)");
    success &= verify(h_src1_B, h_verify0_B, "GPU1[SrcB] -> GPU0[DstB] (on stream2)");

    // --- 11. Cleanup ---
    std::cout << "Cleaning up resources..." << std::endl;
    // Destroy streams (must be done on the context they were created on)
    gpuErrchk(cudaSetDevice(dev0));
    gpuErrchk(cudaStreamDestroy(stream1));
    gpuErrchk(cudaStreamDestroy(stream2));

    // Free device memory (must be done on the context the memory belongs to)
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

    // Disabling peer access is optional, as it's typically reset when the context is destroyed.
    // If needed:
    // gpuErrchk(cudaSetDevice(dev0));
    // gpuErrchk(cudaDeviceDisablePeerAccess(dev1));
    // gpuErrchk(cudaSetDevice(dev1));
    // gpuErrchk(cudaDeviceDisablePeerAccess(dev0));

    // Resetting devices is also optional but can be good practice in some environments.
    // gpuErrchk(cudaSetDevice(dev0));
    // gpuErrchk(cudaDeviceReset());
    // gpuErrchk(cudaSetDevice(dev1));
    // gpuErrchk(cudaDeviceReset());

    std::cout << "\n--- Execution Summary ---" << std::endl;
    if (success) {
        std::cout << "All peer-to-peer copies completed and verified successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cerr << "One or more peer-to-peer copies failed verification." << std::endl;
        return 1; // Indicate failure
    }
}