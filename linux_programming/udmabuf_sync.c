#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* Standard UAPI Headers */
#include <linux/udmabuf.h>
#include <linux/dma-buf.h>
#include <linux/sync_file.h>

int main() {
    int memfd, udbuf_dev, buf_fd, fence_fd;
    size_t buf_size = 4096;

    // 1. Create a sealed memfd (Requirement for udmabuf)
    memfd = memfd_create("sync_demo_buffer", MFD_ALLOW_SEALING);
    if (memfd < 0) {
        perror("memfd_create");
        return 1;
    }
    ftruncate(memfd, buf_size);
    fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL);

    // 2. Create the udmabuf (The DMA-BUF object)
    udbuf_dev = open("/dev/udmabuf", O_RDWR);
    if (udbuf_dev < 0) {
        perror("open /dev/udmabuf (Are you root / is module loaded?)");
        return 1;
    }

    struct udmabuf_create create_arg = {
        .memfd  = memfd,
        .flags  = UDMABUF_FLAGS_CLOEXEC,
        .offset = 0,
        .size   = buf_size,
    };

    buf_fd = ioctl(udbuf_dev, UDMABUF_CREATE, &create_arg);
    if (buf_fd < 0) {
        perror("UDMABUF_CREATE");
        return 1;
    }

    // 3. Export a Sync File (Fence) from the DMA-BUF
    // Note: In a real app, a hardware driver would have attached a fence.
    // If no fence is pending, the kernel returns a 'dummy' signaled fence.
    struct dma_buf_export_sync_file export_args = {
        .flags = DMA_BUF_SYNC_RW, // Wait for both reads and writes
        .fd = -1,
    };

    if (ioctl(buf_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &export_args) < 0) {
        perror("DMA_BUF_IOCTL_EXPORT_SYNC_FILE");
        return 1;
    }
    fence_fd = export_args.fd;

    // read link name of fence_fd
    char link_path[256];
    char path[256];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fence_fd);
    readlink(path, link_path, sizeof(link_path));
    printf("Fence FD link: %s\n", link_path);
    fsync(fence_fd);

    printf("Obtained fence FD: %d\n", fence_fd);
    system("ls -l /proc/self/fd"); // List FDs for debugging
    char fdinfo_cmd[256];
    snprintf(fdinfo_cmd, sizeof(fdinfo_cmd), "cat /proc/self/fdinfo/%d", fence_fd);
    system(fdinfo_cmd); // List FDs for debugging
    system("cat /proc/self/maps"); // Show memory mappings

    // print pid
    printf("Process PID: %d\n", getpid());

    volatile int count = 0;
    while (true) {
	    count++;
    }

    // 4. Use SYNC_IOC_FILE_INFO to inspect the fence
    // We call it twice: once to get the number of fences, once to get the data.
    struct sync_file_info info = { .num_fences = 0 };
    if (ioctl(fence_fd, SYNC_IOC_FILE_INFO, &info) < 0) {
        perror("SYNC_IOC_FILE_INFO (1)");
    } else {
        // Allocate space for fence details if there are underlying fences
        uint32_t num_fences = info.num_fences;
        size_t alloc_size = sizeof(struct sync_file_info) + 
                            (num_fences * sizeof(struct sync_fence_info));
        struct sync_file_info *full_info = malloc(alloc_size);
        
        memset(full_info, 0, alloc_size);
        full_info->num_fences = num_fences;
        full_info->sync_fence_info = (uintptr_t)(full_info + 1);

        if (ioctl(fence_fd, SYNC_IOC_FILE_INFO, full_info) == 0) {
            printf("Sync File Name: %s\n", full_info->name);
            printf("Overall Status: %d (1=done, 0=active, <0=err)\n", full_info->status);
            
            struct sync_fence_info *fences = (struct sync_fence_info *)(uintptr_t)full_info->sync_fence_info;
            for (uint32_t i = 0; i < num_fences; i++) {
                printf("  Fence[%u] driver: %s, status: %d\n", i, fences[i].driver_name, fences[i].status);
            }
        }
        free(full_info);
    }

    system("ls -l /proc/self/fd"); // List FDs for debugging

    // 5. Wait for the fence using standard poll()
    // This is the correct way to wait for a sync_file to signal.
    printf("Waiting for fence to signal...\n");
    struct pollfd pfd = {
        .fd = fence_fd,
        .events = POLLIN, 
    };

    int ret = poll(&pfd, 1, 1000); // 1000ms timeout
    if (ret > 0) {
        printf("Fence signaled! Buffer is safe to access.\n");
    } else if (ret == 0) {
        printf("Timed out waiting for fence.\n");
    } else {
        perror("poll");
    }

    // Cleanup
    close(fence_fd);
    close(buf_fd);
    close(udbuf_dev);
    close(memfd);
    return 0;
}
