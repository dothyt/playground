#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#include "deny_ebpf_demo.skel.h"

static struct deny_ebpf_demo_bpf *skel = NULL;

void sigint_handler(int signo) {
    if (skel) {
        deny_ebpf_demo_bpf__destroy(skel);
        printf("\nUnloaded BPF program and freed resources.\n");
    }
    exit(0);
}

int main(int argc, char **argv) {
    int err;

    signal(SIGINT, sigint_handler);

    skel = deny_ebpf_demo_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open skeleton\n");
        return 1;
    }

    err = deny_ebpf_demo_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load skeleton: %d\n", err);
        deny_ebpf_demo_bpf__destroy(skel);
        return 1;
    }

    err = deny_ebpf_demo_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "Failed to attach skeleton: %d\n", err);
        deny_ebpf_demo_bpf__destroy(skel);
        return 1;
    }

    printf("BPF LSM program loaded and attached. Press Ctrl‑C to unload.\n");

    while (1)
        pause();

    // will never hit here because of exit in signal handler
    return 0;
}
