#include <iostream>
#include <vector>
#include <numeric> // Required for std::iota
#include <stdexcept> // Required for std::runtime_error
#include <cmath> // Required for ceil
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>      // for fork(), sleep(), getpid()
#include <sys/wait.h>    // for waitpid()
#include <semaphore.h>   // for semaphores
#include <fcntl.h>       // for O_CREAT, O_EXCL
#include <sys/stat.h>    // for mode constants (S_IRUSR, S_IWUSR)
#include <time.h>        // For nanosleep (optional better sleep)

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
        // It's possible data might be null temporarily during complex async operations,
        // though less likely in this specific structure after StreamWaitEvent.
        // Keep the check for robustness.
        if (data != nullptr) {
             data[idx] += increment;
        }
    }
}

/**
 * @brief Verifies if the contents of the actual vector match the expected pattern after multiple operations.
 */
bool verifyAccumulatedKernelResult(const std::vector<int>& original, const std::vector<int>& actual, int total_increment, const std::string& name) {
    if (original.size() != actual.size()) {
        std::cerr << "Verification failed for " << name << ": Size mismatch! Expected "
                  << original.size() << ", Got " << actual.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < original.size(); ++i) {
        int expected_value = original[i] + total_increment;
        if (expected_value != actual[i]) {
            // Print only the first mismatch to avoid flooding the console
            std::cerr << "Verification failed for " << name << " at index " << i
                      << ". Original: " << original[i] << ", Expected after kernel: " << expected_value
                      << ", Got: " << actual[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification successful for " << name << "." << std::endl;
    return true;
}


#define SEM_NAME_1 "/my_sync_sem_1"
#define SEM_NAME_2 "/my_sync_sem_2"

// MODIFIED one_process function
int one_process(int myrank) {
    // --- 1. Initialization and Device Check ---
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) { // Need at least 2 GPUs
        std::cerr << "Error: This example requires at least 2 CUDA-enabled GPUs." << std::endl;
        return 1;
    }

    const int dev_sender = myrank;
    const int dev_receiver = (myrank + 1) % deviceCount; // Use the 'next' GPU, wrapping around if needed
    std::cout << myrank << ":" << "Found " << deviceCount << " CUDA devices. Using GPU "<< dev_sender << " (Sender) and GPU " << dev_receiver << " (Receiver)." << std::endl;

    // --- 2. Enable Peer Access ---
    // Sender needs to enable access TO receiver
    gpuErrchk(cudaSetDevice(dev_sender));
    int canAccessPeerSenderToReceiver;
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeerSenderToReceiver, dev_receiver, dev_sender));
    if (!canAccessPeerSenderToReceiver) {
         std::cerr << "Error: Peer access setup failed. GPU " << dev_sender
                   << " cannot access GPU " << dev_receiver << "." << std::endl;
         return 1;
    }
    gpuErrchk(cudaDeviceEnablePeerAccess(dev_receiver, 0));
    std::cout << myrank << ":" << "Enabled peer access for GPU " << dev_sender << " to access GPU " << dev_receiver << "." << std::endl;

    // Receiver needs to enable access TO sender (for events, stream waits potentially)
    gpuErrchk(cudaSetDevice(dev_receiver));
     int canAccessPeerReceiverToSender;
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeerReceiverToSender, dev_sender, dev_receiver));
     if (!canAccessPeerReceiverToSender) {
         std::cerr << "Error: Peer access setup failed. GPU " << dev_receiver
                   << " cannot access GPU " << dev_sender << "." << std::endl;
         // Optional: Disable the previously enabled peer access before returning
         gpuErrchk(cudaSetDevice(dev_sender));
         gpuErrchk(cudaDeviceDisablePeerAccess(dev_receiver));
         return 1;
    }
    gpuErrchk(cudaDeviceEnablePeerAccess(dev_sender, 0));
    std::cout << myrank << ":" << "Enabled peer access for GPU " << dev_receiver << " to access GPU " << dev_sender << "." << std::endl;


    // --- 3. Define Buffer Size and Allocate Host Memory ---
    const size_t N = 1024 * 1024 * 5; // Reduced size for faster testing with 100 iterations
    const size_t bufferSize = N * sizeof(int);
    const int NUM_REQUESTS = 1000;
    std::cout << myrank << ":" << "Buffer size per allocation: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << myrank << ":" << "Number of requests: " << NUM_REQUESTS << std::endl;

    std::vector<int> h_src_A_orig(N), h_src_B_orig(N); // Keep original host data
    std::vector<int> h_verify_A(N), h_verify_B(N);     // For final verification

    std::cout << myrank << ":" << "Initializing host data for sender (GPU " << dev_sender << ")..." << std::endl;
    std::iota(h_src_A_orig.begin(), h_src_A_orig.end(), 1000);      // Pattern A
    std::iota(h_src_B_orig.begin(), h_src_B_orig.end(), 2000000);   // Pattern B

    // --- 4. Allocate Device Memory ---
    int *d_src_A = nullptr, *d_src_B = nullptr; // Sender Buffers
    int *d_dst_A = nullptr, *d_dst_B = nullptr; // Receiver Buffers

    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaMalloc(&d_src_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src_B, bufferSize));
    std::cout << myrank << ":" << "Allocated source buffers on Sender (GPU " << dev_sender << ")." << std::endl;

    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaMalloc(&d_dst_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst_B, bufferSize));
    std::cout << myrank << ":" << "Allocated destination buffers on Receiver (GPU " << dev_receiver << ")." << std::endl;

    // --- 5. Create CUDA Streams and Events ---
    cudaStream_t stream_send_A, stream_send_B;         // Streams for sender copies
    cudaStream_t stream_compute_A, stream_compute_B; // Streams for receiver compute tasks
    cudaStream_t stream_precompute_A, stream_precompute_B; // Streams for precompute tasks
    cudaEvent_t event_A_precompute, event_B_precompute; // Events to signal precompute completion
    cudaEvent_t event_A_copied, event_B_copied;      // Events to signal copy completion
    cudaEvent_t event_A_computed, event_B_computed;  // Events to signal kernel completion

    // Create sender streams & events
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaStreamCreate(&stream_send_A));
    gpuErrchk(cudaStreamCreate(&stream_send_B));
    // Events need to be usable across devices with peer access enabled
    gpuErrchk(cudaEventCreateWithFlags(&event_A_copied, cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&event_B_copied, cudaEventDisableTiming));

    gpuErrchk(cudaStreamCreate(&stream_precompute_A));
    gpuErrchk(cudaStreamCreate(&stream_precompute_B));
    gpuErrchk(cudaEventCreateWithFlags(&event_A_precompute, cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&event_B_precompute, cudaEventDisableTiming));

    // Create receiver compute streams
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaStreamCreate(&stream_compute_A));
    gpuErrchk(cudaStreamCreate(&stream_compute_B));
    // Events need to be usable across devices with peer access enabled
    gpuErrchk(cudaEventCreateWithFlags(&event_A_computed, cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&event_B_computed, cudaEventDisableTiming));
    std::cout << myrank << ":" << "Created streams (send: A, B on GPU"<<dev_sender<<"; compute: A, B on GPU"<<dev_receiver<<") and interprocess events." << std::endl;

    // --- 6. Copy Initial Data from Host to Sender Device ---
    std::cout << myrank << ":" << "Copying initial data from Host to Sender (GPU " << dev_sender << ")..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_sender));
    // Use the specific streams for initial copy as well to ensure ordering if needed later
    gpuErrchk(cudaMemcpyAsync(d_src_A, h_src_A_orig.data(), bufferSize, cudaMemcpyHostToDevice, stream_send_A));
    gpuErrchk(cudaMemcpyAsync(d_src_B, h_src_B_orig.data(), bufferSize, cudaMemcpyHostToDevice, stream_send_B));
    // Synchronize sender streams after initial H2D copy
    gpuErrchk(cudaStreamSynchronize(stream_send_A));
    gpuErrchk(cudaStreamSynchronize(stream_send_B));
    std::cout << myrank << ":" << "Initial data is ready on Sender (GPU " << dev_sender << ")." << std::endl;


    // --- 7. & 8. Loop for Sending Requests (Copy+Event) and Receiving (Wait+Kernel) ---
    std::cout << myrank << ":" << "Issuing " << NUM_REQUESTS << " requests (P2P Copy -> Event -> Wait -> Kernel)..." << std::endl;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < NUM_REQUESTS; ++i) {
        if (i == 0) {
            gpuErrchk(cudaEventRecord(event_A_computed, stream_compute_A));
        } else if (i == 1) {
            gpuErrchk(cudaEventRecord(event_B_computed, stream_compute_B));
        } else if (i % 2 == 0) { // Use Path A (Stream A, Event A, Buffer A)
            gpuErrchk(cudaStreamWaitEvent(stream_precompute_A, event_A_copied, 0));
        } else { // Use Path B (Stream B, Event B, Buffer B)
            gpuErrchk(cudaStreamWaitEvent(stream_precompute_B, event_B_copied, 0));
        }

        if (i % 2 == 0) { // Use Path A (Stream A, Event A, Buffer A)
            // Sender Actions (GPU dev_sender)
            gpuErrchk(cudaSetDevice(dev_sender));

            processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_precompute_A>>>(d_src_A, N, 1); // Increment by 1
            gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
            gpuErrchk(cudaEventRecord(event_A_precompute, stream_precompute_A));
            
            gpuErrchk(cudaStreamWaitEvent(stream_send_A, event_A_precompute, 0));
            //std::cout << myrank << ":" << "  Req " << i << " (A): Enqueue P2P Copy A -> A on stream_send_A" << std::endl;
            gpuErrchk(cudaMemcpyPeerAsync(d_dst_A, dev_receiver, d_src_A, dev_sender, bufferSize, stream_send_A));
            //std::cout << myrank << ":" << "  Req " << i << " (A): Record event_A_copied on stream_send_A" << std::endl;
            gpuErrchk(cudaEventRecord(event_A_copied, stream_send_A));

            // Receiver Actions (GPU dev_receiver)
            gpuErrchk(cudaSetDevice(dev_receiver));
            //std::cout << myrank << ":" << "  Req " << i << " (A): Enqueue Wait for event_A_copied on stream_compute_A" << std::endl;
            gpuErrchk(cudaStreamWaitEvent(stream_compute_A, event_A_copied, 0));
            //std::cout << myrank << ":" << "  Req " << i << " (A): Launch Kernel(+1) on stream_compute_A" << std::endl;
            processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute_A>>>(d_dst_A, N, 1); // Increment by 1
            gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
            gpuErrchk(cudaEventRecord(event_A_computed, stream_compute_A));

        } else { // Use Path B (Stream B, Event B, Buffer B)
            // Sender Actions (GPU dev_sender)
            gpuErrchk(cudaSetDevice(dev_sender));

            processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_precompute_B>>>(d_src_B, N, 2); // Increment by 2
            gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
            gpuErrchk(cudaEventRecord(event_B_precompute, stream_precompute_B));
            
            gpuErrchk(cudaStreamWaitEvent(stream_send_B, event_B_precompute, 0));
            //std::cout << myrank << ":" << "  Req " << i << " (B): Enqueue P2P Copy B -> B on stream_send_B" << std::endl;
            gpuErrchk(cudaMemcpyPeerAsync(d_dst_B, dev_receiver, d_src_B, dev_sender, bufferSize, stream_send_B));
            //std::cout << myrank << ":" << "  Req " << i << " (B): Record event_B_copied on stream_send_B" << std::endl;
            gpuErrchk(cudaEventRecord(event_B_copied, stream_send_B));

            // Receiver Actions (GPU dev_receiver)
            gpuErrchk(cudaSetDevice(dev_receiver));
            //std::cout << myrank << ":" << "  Req " << i << " (B): Enqueue Wait for event_B_copied on stream_compute_B" << std::endl;
            gpuErrchk(cudaStreamWaitEvent(stream_compute_B, event_B_copied, 0));
            //std::cout << myrank << ":" << "  Req " << i << " (B): Launch Kernel(+2) on stream_compute_B" << std::endl;
            processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute_B>>>(d_dst_B, N, 2); // Increment by 2
            gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
            gpuErrchk(cudaEventRecord(event_B_computed, stream_compute_B));
        }
         if (i % 10 == 9) { // Optional: Print progress
             printf("\r%d: Issued %d/%d requests...", myrank, i + 1, NUM_REQUESTS);
             fflush(stdout);
         }
    }
    printf("\n"); // Newline after progress indicator
    std::cout << myrank << ":" << "All " << NUM_REQUESTS << " request commands issued." << std::endl;


    // --- 9. Synchronize Receiver's Compute Streams (AFTER all requests) ---
    std::cout << myrank << ":" << "Synchronizing receiver's compute streams (A and B) AFTER the loop..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver)); // Ensure context is on receiver
    gpuErrchk(cudaStreamSynchronize(stream_compute_A));
    gpuErrchk(cudaStreamSynchronize(stream_compute_B));
    std::cout << myrank << ":" << "Receiver's compute streams synchronized. All kernels should be finished." << std::endl;

    // Optional: Synchronize sender streams as well, although likely less critical here
    // gpuErrchk(cudaSetDevice(dev_sender));
    // gpuErrchk(cudaStreamSynchronize(stream_send_A));
    // gpuErrchk(cudaStreamSynchronize(stream_send_B));
    // std::cout << myrank << ":" << "Sender's send streams synchronized." << std::endl;


    // --- 10. Copy Results Back to Host for Verification ---
    std::cout << myrank << ":" << "Copying final results from Receiver (GPU " << dev_receiver << ") destination buffers back to Host..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver));
    // Use default stream (0) for final blocking copy, or could reuse compute streams
    gpuErrchk(cudaMemcpy(h_verify_A.data(), d_dst_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify_B.data(), d_dst_B, bufferSize, cudaMemcpyDeviceToHost));
    // Explicit device sync after D2H copies might be redundant but ensures completion before verification
    gpuErrchk(cudaDeviceSynchronize());


    // --- 11. Verification ---
    std::cout << myrank << ":" << "Verifying final accumulated results..." << std::endl;
    bool success = true;
    // Buffer A was processed NUM_REQUESTS / 2 times with increment 1
    int total_increment_A =  1 * 2;
    // Buffer B was processed NUM_REQUESTS / 2 times with increment 2
    // (Assumes NUM_REQUESTS is even, adjust if odd needed)
    int total_increment_B =  2 * 2;

    success &= verifyAccumulatedKernelResult(h_src_A_orig, h_verify_A, total_increment_A, "GPU_Receiver[DstA] after " + std::to_string(NUM_REQUESTS/2) + " ops(+1)");
    success &= verifyAccumulatedKernelResult(h_src_B_orig, h_verify_B, total_increment_B, "GPU_Receiver[DstB] after " + std::to_string(NUM_REQUESTS/2) + " ops(+2)");

    // --- 12. Cleanup ---
    std::cout << myrank << ":" << "Cleaning up resources..." << std::endl;
    // Destroy events (associated with sender context)
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaEventDestroy(event_A_copied));
    gpuErrchk(cudaEventDestroy(event_B_copied));

    // Destroy sender streams
    // gpuErrchk(cudaSetDevice(dev_sender)); // Already set
    gpuErrchk(cudaStreamDestroy(stream_send_A));
    gpuErrchk(cudaStreamDestroy(stream_send_B));

    // Destroy receiver streams
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaStreamDestroy(stream_compute_A));
    gpuErrchk(cudaStreamDestroy(stream_compute_B));

    // Free device memory
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaFree(d_src_A));
    gpuErrchk(cudaFree(d_src_B));
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaFree(d_dst_A));
    gpuErrchk(cudaFree(d_dst_B));

    // Disable peer access
    // Sender disables access TO receiver
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaDeviceDisablePeerAccess(dev_receiver));
    // Receiver disables access TO sender
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaDeviceDisablePeerAccess(dev_sender));


    std::cout << myrank << ":" << "--- Execution Summary ---" << std::endl;
    if (success) {
        std::cout << myrank << ":" << "All operations completed and verified successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cerr << myrank << ":" << "One or more steps failed verification." << std::endl;
        return 1; // Indicate failure
    }
}

// Keep the main function as it was, it already calls one_process for two ranks
int main() {
    sem_t *sem1 = SEM_FAILED;
    sem_t *sem2 = SEM_FAILED;
    pid_t pid;

    // --- Cleanup potentially leftover semaphores ---
    sem_unlink(SEM_NAME_1);
    sem_unlink(SEM_NAME_2);

    // --- Create the named semaphores ---
    sem1 = sem_open(SEM_NAME_1, O_CREAT, 0666, 0);
    if (sem1 == SEM_FAILED) {
        perror("Parent: sem_open(SEM_NAME_1) failed");
        exit(EXIT_FAILURE);
    }
    sem2 = sem_open(SEM_NAME_2, O_CREAT, 0666, 0);
    if (sem2 == SEM_FAILED) {
        perror("Parent: sem_open(SEM_NAME_2) failed");
        sem_close(sem1);
        sem_unlink(SEM_NAME_1);
        exit(EXIT_FAILURE);
    }
    printf("Parent [PID %d]: Semaphores created/opened.\n", getpid());

    // --- Fork ---
    pid = fork();
    if (pid < 0) {
        perror("fork failed");
        sem_close(sem1); sem_close(sem2);
        sem_unlink(SEM_NAME_1); sem_unlink(SEM_NAME_2);
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // --- Child Process (Rank 0) ---
        printf("Child  [PID %d, Rank 0]: Started.\n", getpid());
        printf("Child  [PID %d, Rank 0]: Doing setup...\n", getpid());
        sleep(1);
        printf("Child  [PID %d, Rank 0]: Ready. Signaling semaphore 1.\n", getpid());
        if (sem_post(sem1) == -1) { perror("Child: sem_post(sem1) failed"); exit(EXIT_FAILURE); }
        printf("Child  [PID %d, Rank 0]: Waiting for parent (on semaphore 2)...\n", getpid());
        if (sem_wait(sem2) == -1) { perror("Child: sem_wait(sem2) failed"); exit(EXIT_FAILURE); }
        printf("Child  [PID %d, Rank 0]: Parent signaled! Starting one_process(0).\n", getpid());

        int result = 0;
        result = one_process(0); // Child is rank 0

        sem_close(sem1); sem_close(sem2);
        printf("Child  [PID %d, Rank 0]: Exiting with status %d.\n", getpid(), result);
        exit(result); // Exit with the result code from one_process
    } else {
        // --- Parent Process (Rank 1) ---
        printf("Parent [PID %d, Rank 1]: Child process created (PID %d).\n", getpid(), pid);
        printf("Parent [PID %d, Rank 1]: Doing setup...\n", getpid());
        sleep(2);
        printf("Parent [PID %d, Rank 1]: Ready. Signaling semaphore 2.\n", getpid());
        if (sem_post(sem2) == -1) { perror("Parent: sem_post(sem2) failed"); exit(EXIT_FAILURE); }
        printf("Parent [PID %d, Rank 1]: Waiting for child (on semaphore 1)...\n", getpid());
        if (sem_wait(sem1) == -1) { perror("Parent: sem_wait(sem1) failed"); exit(EXIT_FAILURE); }
        printf("Parent [PID %d, Rank 1]: Child signaled! Starting one_process(1).\n", getpid());

        int result = one_process(1); // Parent is rank 1

        // Wait for the child process to finish
        printf("Parent [PID %d, Rank 1]: Waiting for child process %d to exit...\n", getpid(), pid);
        int child_status;
        waitpid(pid, &child_status, 0);
        printf("Parent [PID %d, Rank 1]: Child process finished with status %d.\n", getpid(), WEXITSTATUS(child_status));

        sem_close(sem1); sem_close(sem2);
        printf("Parent [PID %d, Rank 1]: Unlinking semaphores.\n", getpid());
        sem_unlink(SEM_NAME_1); sem_unlink(SEM_NAME_2);

        printf("Parent [PID %d, Rank 1]: Exiting with status %d.\n", getpid(), result);
        exit(result); // Exit with the result code from one_process
    }
    return 1; // Should not be reached
}
