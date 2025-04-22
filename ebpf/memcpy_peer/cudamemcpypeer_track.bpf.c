// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// Structure to hold event data sent to user space
struct event {
    u32 pid;
    u32 tid;
    int dst_device; // Matches cudaMemcpyPeerAsync signature
    int src_device; // Matches cudaMemcpyPeerAsync signature
    u64 count;      // Use u64 for size_t
    u64 stream;     // Use u64 to store the cudaStream_t pointer value
    u64 ts_ns;
};

// Ring buffer map definition
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024); // 256 KB
} events SEC(".maps");

// Attach to the entry of cudaMemcpyPeerAsync
// Note: The actual binary/library path is set by user-space using attach options
SEC("uprobe")
int BPF_KPROBE(handle_memcpy_peer_entry, void *dst, int dstDevice, const void *src, int srcDevice, size_t count, void* stream /* cudaStream_t is void* */) {
    struct event *e;
    u64 id = bpf_get_current_pid_tgid();
    u32 tgid = id >> 32; // Process ID
    u32 tid = (u32)id;    // Thread ID

    // Reserve space in the ring buffer
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        // Cannot print reliably here, ring buffer might be full
        // bpf_printk("Failed to reserve space in ring buffer\n");
        return 0;
    }

    // Populate the event data from function arguments
    e->pid = tgid;
    e->tid = tid;
    e->dst_device = dstDevice;
    e->src_device = srcDevice;
    e->count = (u64)count;
    e->stream = (u64)stream; // Capture the stream pointer value
    e->ts_ns = bpf_ktime_get_ns();

    // Submit the event to user space
    bpf_ringbuf_submit(e, 0);

    return 0;
}