#include <iostream>
#include <vector>
#include <numeric> // Required for std::iota
#include <stdexcept> // Required for std::runtime_error
#include <cmath> // Required for ceil
#include <cuda_runtime.h>

/**
 * @brief CUDA error checking macro.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * @brief Compute kernel executed on GPU 1 - processes a single buffer.
 * @param data Pointer to the device buffer on GPU 1 to process.
 * @param n Number of elements in the buffer.
 * @param increment Value to add to each element.
 */
__global__ void processSingleBufferKernel(int* data, size_t n, int increment) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data != nullptr) {
             data[idx] += increment;
        }
    }
}

/**
 * @brief Verifies if the contents of the actual vector match the expected pattern.
 */
bool verifyKernelResult(const std::vector<int>& original, const std::vector<int>& actual, int increment, const std::string& name) {
    if (original.size() != actual.size()) {
        std::cerr << "Verification failed for " << name << ": Size mismatch! Expected "
                  << original.size() << ", Got " << actual.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < original.size(); ++i) {
        int expected_value = original[i] + increment;
        if (expected_value != actual[i]) {
            std::cerr << "Verification failed for " << name << " at index " << i
                      << ". Original: " << original[i] << ", Expected after kernel: " << expected_value
                      << ", Got: " << actual[i] << std::endl;
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
        return 1;
    }
    std::cout << "Found " << deviceCount << " CUDA devices. Using GPU 0 (Sender) and GPU 1 (Receiver)." << std::endl;
    const int dev_sender = 0;
    const int dev_receiver = 1;

    // --- 2. Enable Peer Access ---
    int canAccessPeer;
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeer, dev_receiver, dev_sender));
     if (!canAccessPeer) {
         std::cerr << "Error: Peer access is not supported. GPU " << dev_receiver
                   << " cannot access GPU " << dev_sender << "." << std::endl;
         return 1;
    }
    std::cout << "Peer access supported (GPU " << dev_receiver << " can access GPU " << dev_sender << ")." << std::endl;

    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaDeviceEnablePeerAccess(dev_receiver, 0));
    std::cout << "Enabled peer access for GPU " << dev_sender << " to write to GPU " << dev_receiver << "'s memory." << std::endl;

    // --- 3. Define Buffer Size and Allocate Host Memory ---
    const size_t N = 1024 * 1024 * 20; // 20 Million integers
    const size_t bufferSize = N * sizeof(int);
    std::cout << "Buffer size per allocation: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;

    std::vector<int> h_src0_A(N), h_src0_B(N);
    std::vector<int> h_verify1_A(N), h_verify1_B(N);

    std::cout << "Initializing host data for sender (GPU 0)..." << std::endl;
    std::iota(h_src0_A.begin(), h_src0_A.end(), 1000);      // Pattern A
    std::iota(h_src0_B.begin(), h_src0_B.end(), 2000000);   // Pattern B

    // --- 4. Allocate Device Memory ---
    int *d_src0_A = nullptr, *d_src0_B = nullptr; // Sender (GPU 0)
    int *d_dst1_A = nullptr, *d_dst1_B = nullptr; // Receiver (GPU 1)

    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaMalloc(&d_src0_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src0_B, bufferSize));
    std::cout << "Allocated source buffers on Sender (GPU " << dev_sender << ")." << std::endl;

    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaMalloc(&d_dst1_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst1_B, bufferSize));
    std::cout << "Allocated destination buffers on Receiver (GPU " << dev_receiver << ")." << std::endl;

    // --- 5. Create CUDA Streams and Events ---
    cudaStream_t stream_A, stream_B;         // Streams for sender copies
    cudaStream_t stream_compute_A, stream_compute_B; // Streams for receiver compute tasks << MODIFIED
    cudaEvent_t event_A_done, event_B_done;   // Events to signal copy completion

    // Create sender streams & events
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaStreamCreate(&stream_A));
    gpuErrchk(cudaStreamCreate(&stream_B));
    gpuErrchk(cudaEventCreate(&event_A_done));
    gpuErrchk(cudaEventCreate(&event_B_done));

    // Create receiver compute streams << MODIFIED
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaStreamCreate(&stream_compute_A));
    gpuErrchk(cudaStreamCreate(&stream_compute_B));
    std::cout << "Created streams (send: A, B on GPU"<<dev_sender<<"; compute: A, B on GPU"<<dev_receiver<<") and events." << std::endl;

    // --- 6. Copy Initial Data from Host to Sender Device ---
    std::cout << "Copying initial data from Host to Sender (GPU 0)..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaMemcpyAsync(d_src0_A, h_src0_A.data(), bufferSize, cudaMemcpyHostToDevice, stream_A));
    gpuErrchk(cudaMemcpyAsync(d_src0_B, h_src0_B.data(), bufferSize, cudaMemcpyHostToDevice, stream_B));
    gpuErrchk(cudaStreamSynchronize(stream_A));
    gpuErrchk(cudaStreamSynchronize(stream_B));
    std::cout << "Initial data is ready on Sender (GPU 0)." << std::endl;

    // --- 7. Sender (GPU 0) Actions: Initiate Peer Copies and Record Events ---
    std::cout << "Sender (GPU 0) initiating asynchronous P2P copies and recording events..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_sender));

    // Copy A -> Dst A on Stream A, then record event A
    std::cout << "  Enqueueing on Stream A: GPU0::d_src0_A -> GPU1::d_dst1_A" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_A, dev_receiver, d_src0_A, dev_sender, bufferSize, stream_A));
    std::cout << "  Recording event_A_done on Stream A" << std::endl;
    gpuErrchk(cudaEventRecord(event_A_done, stream_A));

    // Copy B -> Dst B on Stream B, then record event B
    std::cout << "  Enqueueing on Stream B: GPU0::d_src0_B -> GPU1::d_dst1_B" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_B, dev_receiver, d_src0_B, dev_sender, bufferSize, stream_B));
    std::cout << "  Recording event_B_done on Stream B" << std::endl;
    gpuErrchk(cudaEventRecord(event_B_done, stream_B));

    std::cout << "Sender (GPU 0) P2P copy and event record commands issued." << std::endl;

    // --- 8. Receiver (GPU 1) Actions: Wait for Events Independently and Launch Kernels --- << MODIFIED SECTION
    std::cout << "Receiver (GPU 1) setting up independent waits and compute kernels..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver)); // Ensure context is on receiver

    // --- Path A: Wait for Event A, then process Buffer A ---
    std::cout << "  Path A: Enqueueing wait on stream_compute_A for event_A_done" << std::endl;
    gpuErrchk(cudaStreamWaitEvent(stream_compute_A, event_A_done, 0));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "  Path A: Launching processSingleBufferKernel(+1) for Buffer A on stream_compute_A" << std::endl;
    processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute_A>>>(d_dst1_A, N, 1); // Increment by 1
    gpuErrchk(cudaGetLastError());

    // --- Path B: Wait for Event B, then process Buffer B ---
    std::cout << "  Path B: Enqueueing wait on stream_compute_B for event_B_done" << std::endl;
    gpuErrchk(cudaStreamWaitEvent(stream_compute_B, event_B_done, 0));

    std::cout << "  Path B: Launching processSingleBufferKernel(+2) for Buffer B on stream_compute_B" << std::endl;
    processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute_B>>>(d_dst1_B, N, 2); // Increment by 2
    gpuErrchk(cudaGetLastError());

    std::cout << "Receiver (GPU 1) event waits and kernel launch commands issued independently on compute streams A and B." << std::endl;


    // --- 9. Synchronize Receiver's Compute Streams --- << MODIFIED SECTION
    // Wait for BOTH compute streams on the receiver to finish their work.
    std::cout << "Synchronizing receiver's compute streams (A and B)..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver)); // Ensure context
    gpuErrchk(cudaStreamSynchronize(stream_compute_A));
    gpuErrchk(cudaStreamSynchronize(stream_compute_B));
    std::cout << "Receiver's compute streams synchronized. Kernels should be finished." << std::endl;


    // --- 10. Copy Results Back to Host for Verification ---
    std::cout << "Copying results from Receiver (GPU 1) destination buffers back to Host..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaMemcpy(h_verify1_A.data(), d_dst1_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify1_B.data(), d_dst1_B, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());


    // --- 11. Verification ---
    std::cout << "Verifying copied and processed data..." << std::endl;
    bool success = true;
    success &= verifyKernelResult(h_src0_A, h_verify1_A, 1, "GPU0[SrcA] -> GPU1[DstA] + KernelA(+1)");
    success &= verifyKernelResult(h_src0_B, h_verify1_B, 2, "GPU0[SrcB] -> GPU1[DstB] + KernelB(+2)");

    // --- 12. Cleanup --- << MODIFIED SECTION
    std::cout << "Cleaning up resources..." << std::endl;
    // Destroy events
    gpuErrchk(cudaSetDevice(dev_sender)); // Events created on sender context
    gpuErrchk(cudaEventDestroy(event_A_done));
    gpuErrchk(cudaEventDestroy(event_B_done));

    // Destroy sender streams
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaStreamDestroy(stream_A));
    gpuErrchk(cudaStreamDestroy(stream_B));

    // Destroy receiver streams << MODIFIED
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaStreamDestroy(stream_compute_A));
    gpuErrchk(cudaStreamDestroy(stream_compute_B));

    // Free device memory
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaFree(d_src0_A));
    gpuErrchk(cudaFree(d_src0_B));
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaFree(d_dst1_A));
    gpuErrchk(cudaFree(d_dst1_B));

    std::cout << "\n--- Execution Summary ---" << std::endl;
    if (success) {
        std::cout << "All operations completed and verified successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cerr << "One or more steps failed verification." << std::endl;
        return 1; // Indicate failure
    }
}