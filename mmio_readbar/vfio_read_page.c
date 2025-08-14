#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/vfio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h> // For isprint() in hexdump

// Replace with your device's details
#define IOMMU_GROUP "9"
#define PCI_ADDR "0000:43:00.0"

// Define a standard page size
#define PAGE_SIZE (4096 * 2)

/**
 * hexdump() - Prints a memory region in a classic hexdump format.
 * @ptr: Pointer to the memory to dump.
 * @len: Number of bytes to dump.
 */
void hexdump(const void *ptr, size_t len) {
    const unsigned char *data = (const unsigned char *)ptr;
    size_t i, j;

    for (i = 0; i < len; i += 16) {
        // Print offset
        printf("%08lx: ", i);

        // Print 16 bytes in hex
        for (j = 0; j < 16; j++) {
            if (i + j < len) {
                printf("%02x ", data[i + j]);
            } else {
                printf("   "); // Pad for alignment
            }
        }

        // Print ASCII representation
        printf(" |");
        for (j = 0; j < 16; j++) {
            if (i + j < len) {
                printf("%c", isprint(data[i + j]) ? data[i + j] : '.');
            }
        }
        printf("|\n");
    }
}


int main() {
    int container, group, device;
    char path[128];

    // 1. Open the VFIO container
    container = open("/dev/vfio/vfio", O_RDWR);
    if (container < 0) {
        perror("Failed to open /dev/vfio/vfio");
        return 1;
    }

    // ... (VFIO setup for group and container is identical to before) ...
    snprintf(path, sizeof(path), "/dev/vfio/%s", IOMMU_GROUP);
    group = open(path, O_RDWR);
    if (group < 0) { perror("Failed to open IOMMU group file"); close(container); return 1; }

    if (ioctl(group, VFIO_GROUP_SET_CONTAINER, &container) < 0) { perror("Failed to set container for the group"); close(group); close(container); return 1; }
    if (ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) < 0) { perror("Failed to set IOMMU type"); close(group); close(container); return 1; }

    // 5. Get a file descriptor for the device
    device = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, PCI_ADDR);
    if (device < 0) {
        perror("Failed to get device FD");
        close(group);
        close(container);
        return 1;
    }

    printf("Successfully set up VFIO for device %s\n", PCI_ADDR);

    // 6. Get information about BAR0
    struct vfio_region_info region_info = { .argsz = sizeof(region_info) };
    region_info.index = VFIO_PCI_BAR0_REGION_INDEX;
    if (ioctl(device, VFIO_DEVICE_GET_REGION_INFO, &region_info) < 0) {
        perror("Failed to get region info for BAR0");
        close(device); close(group); close(container);
        return 1;
    }

    // 7. Map the BAR0 into user memory
    void *bar0_ptr = mmap(NULL, region_info.size, PROT_READ | PROT_WRITE, MAP_SHARED, device, region_info.offset);
    if (bar0_ptr == MAP_FAILED) {
        perror("Failed to mmap BAR0");
        close(device); close(group); close(container);
        return 1;
    }

    printf("BAR0 mapped at %p with size 0x%llx. Reading first page...\n", bar0_ptr, (unsigned long long)region_info.size);
    printf("-----------------------------------------------------------------\n");

    // 8. Determine how much to read (up to one page)
    size_t bytes_to_read = (region_info.size < PAGE_SIZE) ? region_info.size : PAGE_SIZE;

    // Call the hexdump function to display the memory
    hexdump(bar0_ptr, bytes_to_read);

    printf("-----------------------------------------------------------------\n");

    // 9. Cleanup
    munmap(bar0_ptr, region_info.size);
    close(device);
    close(group);
    close(container);

    printf("Cleanup complete.\n");

    return 0;
}
