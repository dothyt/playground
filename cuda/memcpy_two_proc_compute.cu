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

#define SEM_NAME_1 "/my_sync_sem_1"
#define SEM_NAME_2 "/my_sync_sem_2"

int one_process(int myrank) {
    // --- 1. Initialization and Device Check ---
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount != 2) {
        std::cerr << "Error: This example requires at least 2 CUDA-enabled GPUs." << std::endl;
        return 1;
    }
    
    const int dev_sender = myrank;
    const int dev_receiver = deviceCount - 1 - myrank; // Use the other GPU as receiver
    std::cout << myrank << ":" << "Found " << deviceCount << " CUDA devices. Using GPU "<< dev_sender << " (Sender) and GPU " << dev_receiver << " (Receiver)." << std::endl;

    // --- 2. Enable Peer Access ---
    int canAccessPeer;
    gpuErrchk(cudaDeviceCanAccessPeer(&canAccessPeer, dev_receiver, dev_sender));
     if (!canAccessPeer) {
         std::cerr << "Error: Peer access is not supported. GPU " << dev_receiver
                   << " cannot access GPU " << dev_sender << "." << std::endl;
         return 1;
    }
    std::cout << myrank << ":" << "Peer access supported (GPU " << dev_receiver << " can access GPU " << dev_sender << ")." << std::endl;

    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaDeviceEnablePeerAccess(dev_receiver, 0));
    std::cout << myrank << ":" << "Enabled peer access for GPU " << dev_sender << " to write to GPU " << dev_receiver << "'s memory." << std::endl;

    // --- 3. Define Buffer Size and Allocate Host Memory ---
    const size_t N = 1024 * 1024 * 20; // 20 Million integers
    const size_t bufferSize = N * sizeof(int);
    std::cout << myrank << ":" << "Buffer size per allocation: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;

    std::vector<int> h_src0_A(N), h_src0_B(N);
    std::vector<int> h_verify1_A(N), h_verify1_B(N);

    std::cout << myrank << ":" << "Initializing host data for sender (GPU " << dev_sender << ")..." << std::endl;
    std::iota(h_src0_A.begin(), h_src0_A.end(), 1000);      // Pattern A
    std::iota(h_src0_B.begin(), h_src0_B.end(), 2000000);   // Pattern B

    // --- 4. Allocate Device Memory ---
    int *d_src0_A = nullptr, *d_src0_B = nullptr; // Sender (GPU 0)
    int *d_dst1_A = nullptr, *d_dst1_B = nullptr; // Receiver (GPU 1)

    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaMalloc(&d_src0_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_src0_B, bufferSize));
    std::cout << myrank << ":" << "Allocated source buffers on Sender (GPU " << dev_sender << ")." << std::endl;

    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaMalloc(&d_dst1_A, bufferSize));
    gpuErrchk(cudaMalloc(&d_dst1_B, bufferSize));
    std::cout << myrank << ":" << "Allocated destination buffers on Receiver (GPU " << dev_receiver << ")." << std::endl;

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
    std::cout << myrank << ":" << "Created streams (send: A, B on GPU"<<dev_sender<<"; compute: A, B on GPU"<<dev_receiver<<") and events." << std::endl;

    // --- 6. Copy Initial Data from Host to Sender Device ---
    std::cout << myrank << ":" << "Copying initial data from Host to Sender (GPU " << dev_sender << ")..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_sender));
    gpuErrchk(cudaMemcpyAsync(d_src0_A, h_src0_A.data(), bufferSize, cudaMemcpyHostToDevice, stream_A));
    gpuErrchk(cudaMemcpyAsync(d_src0_B, h_src0_B.data(), bufferSize, cudaMemcpyHostToDevice, stream_B));
    gpuErrchk(cudaStreamSynchronize(stream_A));
    gpuErrchk(cudaStreamSynchronize(stream_B));
    std::cout << myrank << ":" << "Initial data is ready on Sender (GPU " << dev_sender << ")." << std::endl;

    // --- 7. Sender (GPU 0) Actions: Initiate Peer Copies and Record Events ---
    std::cout << myrank << ":" << "Sender (GPU " << dev_sender << ") initiating asynchronous P2P copies and recording events..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_sender));

    // Copy A -> Dst A on Stream A, then record event A
    std::cout << myrank << ":" << "  Enqueueing on Stream A: GPU" << dev_sender << "::d_src0_A -> GPU" << dev_receiver << "::d_dst1_A" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_A, dev_receiver, d_src0_A, dev_sender, bufferSize, stream_A));
    std::cout << myrank << ":" << "  Recording event_A_done on Stream A" << std::endl;
    gpuErrchk(cudaEventRecord(event_A_done, stream_A));

    // Copy B -> Dst B on Stream B, then record event B
    std::cout << myrank << ":" << "  Enqueueing on Stream B: GPU" << dev_sender << "::d_src0_B -> GPU" << dev_receiver << "::d_dst1_B" << std::endl;
    gpuErrchk(cudaMemcpyPeerAsync(d_dst1_B, dev_receiver, d_src0_B, dev_sender, bufferSize, stream_B));
    std::cout << myrank << ":" << "  Recording event_B_done on Stream B" << std::endl;
    gpuErrchk(cudaEventRecord(event_B_done, stream_B));

    std::cout << myrank << ":" << "Sender (GPU " << dev_sender << ") P2P copy and event record commands issued." << std::endl;

    // --- 8. Receiver (GPU 1) Actions: Wait for Events Independently and Launch Kernels --- << MODIFIED SECTION
    std::cout << myrank << ":" << "Receiver (GPU " << dev_receiver << ") setting up independent waits and compute kernels..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver)); // Ensure context is on receiver

    // --- Path A: Wait for Event A, then process Buffer A ---
    std::cout << myrank << ":" << "  Path A: Enqueueing wait on stream_compute_A for event_A_done" << std::endl;
    gpuErrchk(cudaStreamWaitEvent(stream_compute_A, event_A_done, 0));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << myrank << ":" << "  Path A: Launching processSingleBufferKernel(+1) for Buffer A on stream_compute_A" << std::endl;
    processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute_A>>>(d_dst1_A, N, 1); // Increment by 1
    gpuErrchk(cudaGetLastError());

    // --- Path B: Wait for Event B, then process Buffer B ---
    std::cout << myrank << ":" << "  Path B: Enqueueing wait on stream_compute_B for event_B_done" << std::endl;
    gpuErrchk(cudaStreamWaitEvent(stream_compute_B, event_B_done, 0));

    std::cout << myrank << ":" << "  Path B: Launching processSingleBufferKernel(+2) for Buffer B on stream_compute_B" << std::endl;
    processSingleBufferKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute_B>>>(d_dst1_B, N, 2); // Increment by 2
    gpuErrchk(cudaGetLastError());

    std::cout << myrank << ":" << "Receiver (GPU " << dev_receiver << ") event waits and kernel launch commands issued independently on compute streams A and B." << std::endl;


    // --- 9. Synchronize Receiver's Compute Streams --- << MODIFIED SECTION
    // Wait for BOTH compute streams on the receiver to finish their work.
    std::cout << myrank << ":" << "Synchronizing receiver's compute streams (A and B)..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver)); // Ensure context
    gpuErrchk(cudaStreamSynchronize(stream_compute_A));
    gpuErrchk(cudaStreamSynchronize(stream_compute_B));
    std::cout << myrank << ":" << "Receiver's compute streams synchronized. Kernels should be finished." << std::endl;


    // --- 10. Copy Results Back to Host for Verification ---
    std::cout << myrank << ":" << "Copying results from Receiver (GPU " << dev_receiver << ") destination buffers back to Host..." << std::endl;
    gpuErrchk(cudaSetDevice(dev_receiver));
    gpuErrchk(cudaMemcpy(h_verify1_A.data(), d_dst1_A, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_verify1_B.data(), d_dst1_B, bufferSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());


    // --- 11. Verification ---
    std::cout << myrank << ":" << "Verifying copied and processed data..." << std::endl;
    bool success = true;
    success &= verifyKernelResult(h_src0_A, h_verify1_A, 1, "GPU0[SrcA] -> GPU1[DstA] + KernelA(+1)");
    success &= verifyKernelResult(h_src0_B, h_verify1_B, 2, "GPU0[SrcB] -> GPU1[DstB] + KernelB(+2)");

    // --- 12. Cleanup --- << MODIFIED SECTION
    std::cout << myrank << ":" << "Cleaning up resources..." << std::endl;
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

    std::cout << myrank << ":" << "--- Execution Summary ---" << std::endl;
    if (success) {
        std::cout << "All operations completed and verified successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cerr << "One or more steps failed verification." << std::endl;
        return 1; // Indicate failure
    }
}

int main() {
    sem_t *sem1 = SEM_FAILED;
    sem_t *sem2 = SEM_FAILED;
    pid_t pid;

    // --- Cleanup potentially leftover semaphores from previous runs ---
    // It's okay if these fail (e.g., if they don't exist)
    sem_unlink(SEM_NAME_1);
    sem_unlink(SEM_NAME_2);

    // --- Create the named semaphores ---
    // O_CREAT: Create if it doesn't exist
    // 0666: Permissions (read/write for owner, group, others - adjust if needed)
    // 0: Initial value (no process has signaled yet)
    sem1 = sem_open(SEM_NAME_1, O_CREAT, 0666, 0);
    if (sem1 == SEM_FAILED) {
        perror("Parent: sem_open(SEM_NAME_1) failed");
        exit(EXIT_FAILURE);
    }

    sem2 = sem_open(SEM_NAME_2, O_CREAT, 0666, 0);
    if (sem2 == SEM_FAILED) {
        perror("Parent: sem_open(SEM_NAME_2) failed");
        sem_close(sem1); // Clean up the first semaphore
        sem_unlink(SEM_NAME_1);
        exit(EXIT_FAILURE);
    }

    printf("Parent [PID %d]: Semaphores created/opened.\n", getpid());

    // --- Fork to create the second process ---
    pid = fork();
    if (pid < 0) {
        // Fork failed
        perror("fork failed");
        sem_close(sem1);
        sem_close(sem2);
        sem_unlink(SEM_NAME_1);
        sem_unlink(SEM_NAME_2);
        exit(EXIT_FAILURE);

        std::cerr << "Fork failed!" << std::endl;
        return 1;
    } else if (pid == 0) {
        // --- Child Process ---
        printf("Child  [PID %d]: Started.\n", getpid());

        // Simulate child-specific setup
        printf("Child  [PID %d]: Doing setup...\n", getpid());
        sleep(1); // Simulate work

        printf("Child  [PID %d]: Ready. Signaling semaphore 1.\n", getpid());
        if (sem_post(sem1) == -1) { // Signal that child is ready
            perror("Child: sem_post(sem1) failed");
            // Semaphores are closed in the exit handler or automatically by OS
             exit(EXIT_FAILURE); // Exit child on error
        }

        printf("Child  [PID %d]: Waiting for parent (on semaphore 2)...\n", getpid());
        if (sem_wait(sem2) == -1) { // Wait for parent's signal
             perror("Child: sem_wait(sem2) failed");
             exit(EXIT_FAILURE); // Exit child on error
        }

        printf("Child  [PID %d]: Parent signaled! Starting action.\n", getpid());
        int myrank = 0;
        int result = one_process(myrank);

        sem_close(sem1);
        sem_close(sem2);
        printf("Child  [PID %d]: Exiting.\n", getpid());
        exit(EXIT_SUCCESS); // Child exits successfully
    } else {
        // --- Parent Process ---
        printf("Parent [PID %d]: Child process created (PID %d).\n", getpid(), pid);

        // Simulate parent-specific setup
        printf("Parent [PID %d]: Doing setup...\n", getpid());
        sleep(2); // Simulate different amount of work than child

        printf("Parent [PID %d]: Ready. Signaling semaphore 2.\n", getpid());
        if (sem_post(sem2) == -1) { // Signal that parent is ready
            perror("Parent: sem_post(sem2) failed");
            // Consider killing child if parent fails here? Depends on requirements.
             // Clean up and exit parent on error
             sem_close(sem1);
             sem_close(sem2);
             sem_unlink(SEM_NAME_1);
             sem_unlink(SEM_NAME_2);
             // waitpid(pid, NULL, 0); // Wait for child before exiting? Optional.
             exit(EXIT_FAILURE);
        }

        printf("Parent [PID %d]: Waiting for child (on semaphore 1)...\n", getpid());
        if (sem_wait(sem1) == -1) { // Wait for child's signal
             perror("Parent: sem_wait(sem1) failed");
             // Clean up and exit parent on error
             sem_close(sem1);
             sem_close(sem2);
             sem_unlink(SEM_NAME_1);
             sem_unlink(SEM_NAME_2);
             // waitpid(pid, NULL, 0); // Wait for child before exiting? Optional.
             exit(EXIT_FAILURE);
        }

        printf("Parent [PID %d]: Child signaled! Starting action.\n", getpid());
        int myrank = 1;
        int result = one_process(myrank);

        // Wait for the child process to finish
        printf("Parent [PID %d]: Waiting for child process to exit...\n", getpid());
        waitpid(pid, NULL, 0);
        printf("Parent [PID %d]: Child process finished.\n", getpid());

        // Close semaphores (detach parent)
        sem_close(sem1);
        sem_close(sem2);

        // --- Unlink the semaphores ---
        // Only one process needs to do this after both are done.
        // The parent is a natural place after waiting for the child.
        printf("Parent [PID %d]: Unlinking semaphores.\n", getpid());
        sem_unlink(SEM_NAME_1);
        sem_unlink(SEM_NAME_2);

        printf("Parent [PID %d]: Exiting.\n", getpid());
        exit(EXIT_SUCCESS); // Parent exits successfully
    }

    return 1; // This line should never be reached
}