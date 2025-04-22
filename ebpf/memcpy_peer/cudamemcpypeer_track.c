// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <bpf/libbpf.h> // Provides ring_buffer__* functions and other libbpf helpers
#include <bpf/bpf.h>
#include <sys/resource.h>
#include "cudamemcpypeer_track.skel.h" // Generated skeleton header

// Define the event structure (must match the BPF side exactly)
struct event {
    unsigned int pid;
    unsigned int tid;
    int dst_device;
    int src_device;
    unsigned long long count;
    unsigned long long stream; // Will display as hex pointer
    unsigned long long ts_ns;
};

// Global flag for signal handling
static volatile sig_atomic_t exiting = 0;

// Signal handler function
static void sig_handler(int sig) {
    exiting = 1;
}

// Callback function for processing events received from the kernel ring buffer
static int handle_event(void *ctx, void *data, size_t data_sz) {
    const struct event *e = data;
    struct timespec ts_now;
    time_t T;
    struct tm tm_local;

    // Get current time to print alongside the event
    // Note: Kernel timestamp (e->ts_ns) is monotonic, not wall-clock
    clock_gettime(CLOCK_REALTIME, &ts_now);
    T = ts_now.tv_sec;
    localtime_r(&T, &tm_local); // Use re-entrant version

    // Print event details
    printf("%02d:%02d:%02d.%09ld | PID: %-7u | TID: %-7u | cudaMemcpyPeerAsync(dstDev: %d, srcDev: %d, count: %llu, stream: 0x%llx)\n",
           tm_local.tm_hour, tm_local.tm_min, tm_local.tm_sec, ts_now.tv_nsec, // Using user-space time for wall-clock
           e->pid, e->tid, e->dst_device, e->src_device, e->count, e->stream);
           // Alternative: Print kernel monotonic time directly: e->ts_ns

    return 0; // Indicate success processing this event
}

// Helper function to increase the RLIMIT_MEMLOCK resource limit
static int bump_memlock_rlimit(void) {
    struct rlimit rlim_new = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };
    if (setrlimit(RLIMIT_MEMLOCK, &rlim_new)) {
        fprintf(stderr, "Failed to increase RLIMIT_MEMLOCK limit! (%s)\n", strerror(errno));
        fprintf(stderr, "Try running with root privileges (sudo) or setting capabilities.\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    struct cudamemcpypeer_track_bpf *skel = NULL; // BPF skeleton object
    struct ring_buffer *rb = NULL;                // Ring buffer manager object
    int err;
    const char *target_bin = NULL;
    const char *target_func = "cudaMemcpyPeerAsync"; // The CUDA function we are tracing

    // --- Argument Parsing ---
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_binary_or_library>\n", argv[0]);
        fprintf(stderr, "Description: Traces cudaMemcpyPeerAsync calls in the specified target.\n");
        fprintf(stderr, "Example (tracing library): %s /usr/local/cuda/lib64/libcudart.so.11.0\n", argv[0]);
        fprintf(stderr, "Example (tracing app):   %s ./my_peer_memcpy_app\n", argv[0]);
        return 1;
    }
    target_bin = argv[1];

    // --- Setup ---
    // Set up signal handler for graceful shutdown
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Set libbpf print level (useful for debugging BPF issues)
    // LIBBPF_PRINT_LEVEL_DEBUG provides more verbose output if needed
    libbpf_set_print(LIBBPF_WARN);

    // Increase memory lock limit (required for BPF maps)
    if (bump_memlock_rlimit() < 0) {
        return 1;
    }

    // --- BPF Skeleton Management ---
    // Open BPF application skeleton
    skel = cudamemcpypeer_track_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    // Load & verify BPF programs and maps into the kernel
    err = cudamemcpypeer_track_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load and verify BPF skeleton\n");
        goto cleanup;
    }

    // --- Attach BPF Program (Uprobe) ---
    // Prepare options for attaching the uprobe
    // We specify the function name and libbpf will find the offset
    LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts, .func_name = target_func, .retprobe = false);

    // Attach the uprobe to the specified function in the target binary/library
    // - PID -1 means attach globally to any process using this binary/library path.
    // - Offset 0 combined with func_name tells libbpf to resolve the symbol.
    skel->links.handle_memcpy_peer_entry = bpf_program__attach_uprobe_opts(
        skel->progs.handle_memcpy_peer_entry, // The BPF program to attach (from skeleton)
        -1,           // Attach globally (all relevant processes)
        target_bin,   // Path to the target binary/library
        0,            // Offset (0 when using func_name in opts)
        &uprobe_opts); // Options struct defined above

    // Check if attachment was successful
    if (!skel->links.handle_memcpy_peer_entry) {
        err = -errno; // libbpf usually sets errno on failure
        fprintf(stderr, "Failed to attach uprobe to %s:%s (%d: %s)\n",
                target_bin, target_func, err, strerror(-err));
        goto cleanup;
    }

    printf("Successfully attached BPF program. Tracing %s calls in %s...\n", target_func, target_bin);
    printf("Press Ctrl+C to stop.\n");
    printf("-------------------------------------------------------------------------------------------------------------------------\n");
    printf("%-25s | %-11s | %-11s | %s\n", "TIME (Local)", "PID", "TID", "EVENT DETAILS");
    printf("-------------------------------------------------------------------------------------------------------------------------\n");

    // --- Ring Buffer Setup ---
    // Create the ring buffer manager using the map's file descriptor
    // Pass the handle_event callback function to process incoming data
    // *** CORRECTED FUNCTION NAME ***
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) {
        err = -errno; // Ring buffer creation sets errno
        fprintf(stderr, "Failed to create ring buffer (%d: %s)\n", err, strerror(-err));
        goto cleanup;
    }

    // --- Main Event Loop ---
    // Poll the ring buffer for new events until interrupted
    while (!exiting) {
        // *** CORRECTED FUNCTION NAME ***
        // Poll with a timeout (e.g., 100ms). Returns number of events consumed or < 0 on error.
        err = ring_buffer__poll(rb, 100 /* timeout_ms */);

        // Handle EINTR (interruption by signal) gracefully
        if (err == -EINTR) {
            err = 0; // Reset error, loop will check 'exiting' flag
            continue;
        }
        // Handle other polling errors
        if (err < 0) {
            fprintf(stderr, "Error polling ring buffer (%d): %s\n", err, strerror(-err));
            break; // Exit loop on error
        }
        // If err >= 0, handle_event() was called for each received event by libbpf
    }

    fflush(stdout);

// --- Cleanup ---
cleanup:
    printf("\nDetaching and cleaning up...\n");

    // Free the ring buffer manager
    // *** CORRECTED FUNCTION NAME ***
    ring_buffer__free(rb); // Tolerates NULL pointer

    // Destroy the BPF skeleton
    // This automatically detaches probes (links) and unloads programs/maps
    cudamemcpypeer_track_bpf__destroy(skel); // Tolerates NULL pointer

    printf("Cleanup complete.\n");
    return err < 0 ? 1 : 0; // Return non-zero on error
}
