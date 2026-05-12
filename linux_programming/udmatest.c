#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/udmabuf.h>
#include <errno.h>
#include <string.h>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/kcmp.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

int read_fd_link(int lfd, char *buf, size_t size)
{
	char t[32];
	ssize_t ret;

	snprintf(t, sizeof(t), "/proc/self/fd/%d", lfd);
	ret = readlink(t, buf, size);
	if (ret < 0) {
		perror("Can't read link of fd");
		return -1;
	} else if ((size_t)ret >= size) {
		fprintf(stderr, "Buffer for read link of fd is too small\n");
		return -1;
	}
	buf[ret] = 0;

	return ret;
}

int is_same_file(int fd1, int fd2) {
    return syscall(SYS_kcmp, getpid(), getpid(), KCMP_FILE, fd1, fd2);
}

/* Helper: Get physical address (PFN) of a virtual address */
uint64_t get_pfn(uintptr_t virt_addr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open pagemap failed");
        return 0;
    }

    uint64_t entry;
    // Pagemap entries are 8 bytes; offset is (virt_addr / page_size) * 8
    off_t offset = (virt_addr / sysconf(_SC_PAGESIZE)) * 8;
    
    if (pread(fd, &entry, 8, offset) != 8) {
        close(fd);
        return 0;
    }

    close(fd);
    // Bit 63 is "page present". Bits 0-54 represent the PFN.
    printf("Page map entry: %llx\n", entry);
    if (!(entry & (1ULL << 63))) return 0;
    return entry & ((1ULL << 55) - 1);
}

/**
 * verify_page_sharing: 100% verification that two FDs share the same memory.
 */
int verify_page_sharing(int fd1, int fd2) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    
    // 1. Map one page from each
    void *ptr1 = mmap(NULL, page_size, PROT_READ, MAP_SHARED, fd1, 0);
    void *ptr2 = mmap(NULL, page_size, PROT_READ, MAP_SHARED, fd2, 0);
    printf("Mapped addresses: ptr1=%p, ptr2=%p\n", ptr1, ptr2);
    
    if (ptr1 == MAP_FAILED || ptr2 == MAP_FAILED) {
        if (ptr1 != MAP_FAILED) munmap(ptr1, page_size);
        if (ptr2 != MAP_FAILED) munmap(ptr2, page_size);
        return 0;
    }

    // 2. Force the pages into RAM (touch them)
    volatile char dummy;
    dummy = *(char *)ptr1;
    dummy = *(char *)ptr2;

    // 3. Get Physical Frame Numbers
    uint64_t pfn1 = get_pfn((uintptr_t)ptr1);
    uint64_t pfn2 = get_pfn((uintptr_t)ptr2);

    printf("PFN1: %lx, PFN2: %lx\n", pfn1, pfn2);

    munmap(ptr1, page_size);
    munmap(ptr2, page_size);

    return (pfn1 != 0 && pfn1 == pfn2);
}

/* Helper to get the Physical Frame Number (PFN) */
uint64_t get_pfn_robust(void *virt_addr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open pagemap failed");
        return 0;
    }

    uint64_t entry;
    // Pagemap is indexed by virtual page number
    off_t offset = ((uintptr_t)virt_addr / sysconf(_SC_PAGESIZE)) * 8;
    
    if (pread(fd, &entry, 8, offset) != 8) {
        perror("pread pagemap failed");
        close(fd);
        return 0;
    }
    close(fd);

    // Bit 63: Page Present. Bit 62: Page Swapped.
    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "Page not present in RAM, %lx\n", entry);
        return 0;
    }
    
    // Bits 0-54 are the PFN
    return entry & ((1ULL << 55) - 1);
}

/**
 * find_memfd_for_udmabuf: The only 100% way to link them.
 * Iterates through memfds and compares physical backing.
 */
int find_memfd_for_udmabuf(int udma_fd, size_t size) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    
    // 1. Map the udmabuf and FORCE it into RAM
    void *udma_ptr = mmap(NULL, page_size, PROT_READ, MAP_SHARED, udma_fd, 0);
    if (udma_ptr == MAP_FAILED) return -1;
    
    // Force the kernel to allocate/load the physical page
    volatile char dummy = *(volatile char *)udma_ptr;
    printf("dummy read: %x\n", dummy);
    uint64_t udma_pfn = get_pfn_robust(udma_ptr);
    printf("UDMABUF PFN: %lx\n", udma_pfn);

    int result_fd = -1;
    // 2. Iterate through all other FDs to find the memfd
    for (int i = 0; i < 1024; i++) {
        char path[256], link[256];
        snprintf(path, sizeof(path), "/proc/self/fd/%d", i);
        if (readlink(path, link, sizeof(link)-1) <= 0) continue;

        if (strstr(link, "/memfd:")) {
            // Found a memfd, now check if it shares the same physical page
            void *mem_ptr = mmap(NULL, page_size, PROT_READ, MAP_SHARED, i, 0);
            if (mem_ptr == MAP_FAILED) continue;

            volatile char dummy2 = *(volatile char *)mem_ptr;
            if (get_pfn_robust(mem_ptr) == udma_pfn) {
                result_fd = i;
                munmap(mem_ptr, page_size);
                break; 
            }
            munmap(mem_ptr, page_size);
        }
    }

    munmap(udma_ptr, page_size);
    return result_fd;
}

#include <sys/mman.h>
#include <unistd.h>

/**
 * verify_dmabuf_memfd_link: Proves the link by writing a temporary signature.
 */
int verify_dmabuf_memfd_link(int dmabuf_fd, int memfd_fd) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    int match = 0;

    void *dma_ptr = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    void *mem_ptr = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_fd, 0);

    if (dma_ptr != MAP_FAILED && mem_ptr != MAP_FAILED) {
        volatile unsigned char *d_ptr = (unsigned char *)dma_ptr;
        volatile unsigned char *m_ptr = (unsigned char *)mem_ptr;

        unsigned char original = *m_ptr;
        unsigned char signature = 0x55; // Arbitrary pattern

        // If the original happens to be our signature, use a different one
        if (original == signature) signature = 0xAA;

        *d_ptr = signature;     // Write to dmabuf
        printf("*m_ptr = %x, signature = %x\n", *m_ptr, signature);
        if (*m_ptr == signature) { // Check memfd
            match = 1;
        }

        *d_ptr = original;      // Restore original byte
    }

    if (dma_ptr != MAP_FAILED) munmap(dma_ptr, page_size);
    if (mem_ptr != MAP_FAILED) munmap(mem_ptr, page_size);

    return match;
}

#define CANARY_SIZE 8
const uint64_t MAGIC_A = 0xDEADC0DEBAADF00D;
const uint64_t MAGIC_B = 0x2121424284841616; // Inversion or secondary check

/**
 * find_udmabuf_offset: Finds the starting offset of a dmabuf within a memfd.
 * Returns the offset in bytes, or -1 if not found.
 */
off_t find_udmabuf_offset(int dmabuf_fd, int memfd_fd, size_t memfd_size) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    off_t found_offset = -1;

    // 1. Map the start of the dmabuf (The Needle)
    uint64_t *dma_ptr = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    // 2. Map the entire memfd (The Haystack)
    uint64_t *mem_ptr = mmap(NULL, memfd_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_fd, 0);

    if (dma_ptr == MAP_FAILED || mem_ptr == MAP_FAILED) {
        if (dma_ptr != MAP_FAILED) munmap(dma_ptr, page_size);
        if (mem_ptr != MAP_FAILED) munmap(mem_ptr, memfd_size);
        return -1;
    }

    // 3. Save original data to restore later
    uint64_t original_data = dma_ptr[0];

    // 4. Write Canary A
    dma_ptr[0] = MAGIC_A;

    // 5. Scan the haystack (memfd) for the canary
    // We only need to check every page boundary, as udmabuf offsets are page-aligned
    for (off_t off = 0; off <= (off_t)(memfd_size - page_size); off += page_size) {
        uint64_t *candidate = (uint64_t *)((uint8_t *)mem_ptr + off);
        
        if (*candidate == MAGIC_A) {
            // Potential match! Perform Inversion Check to be 100% sure
            dma_ptr[0] = MAGIC_B;
            if (*candidate == MAGIC_B) {
                found_offset = off;
                break;
            }
            // If it failed MAGIC_B, it was a random collision; keep searching
            dma_ptr[0] = MAGIC_A; 
        }
    }

    // 6. Restore original state
    dma_ptr[0] = original_data;

    munmap(dma_ptr, page_size);
    munmap(mem_ptr, memfd_size);

    return found_offset;
}


int main() {
    int memfd, udbfd, dma_buf_fd;
    size_t width = 1920;
    size_t height = 1080;
    size_t size = width * height * 4; // 32-bit RGBA
    unsigned long count = 0;
    struct stat st;

    // print pid
    printf("Process PID: %d\n", getpid());

    // 1. Create a memfd (anonymous shared memory)
    memfd = memfd_create("software-buffer", MFD_ALLOW_SEALING);
    if (memfd < 0) {
        perror("memfd_create");
        return 1;
    }

    // 2. Set the size and seal the memfd
    // Some kernels require seals to ensure the buffer doesn't shrink during DMA
    if (ftruncate(memfd, size) < 0) {
        perror("ftruncate");
        return 1;
    }
    fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL);

    if (fstat(memfd, &st) == -1) {
		perror("fstat error");
		return -1;
	}
    // print every single field in st
    printf("memfd Stat:\n");
    printf("  st_dev: %lu\n", st.st_dev);
    printf("  st_ino: %lu\n", st.st_ino);
    printf("  st_mode: %o\n", st.st_mode);
    printf("  st_nlink: %lu\n", st.st_nlink);
    printf("  st_uid: %u\n", st.st_uid);
    printf("  st_gid: %u\n", st.st_gid);
    printf("  st_rdev: %lu\n", st.st_rdev);
    printf("  st_size: %ld\n", st.st_size);
    printf("  st_blksize: %ld\n", st.st_blksize);
    printf("  st_blocks: %ld\n", st.st_blocks);
    printf("  st_atime: %ld\n", st.st_atime);
    printf("  st_mtime: %ld\n", st.st_mtime);
    printf("  st_ctime: %ld\n", st.st_ctime);

    // 3. Map the memory to "render" something into it (Software Rendering)
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
    printf("Mapped memfd at address: %p\n", ptr);
    // memset(ptr, 0xff, size); // Fill with white (as a dummy render)
    // munmap(ptr, size);

    // 4. Open the udmabuf device
    udbfd = open("/dev/udmabuf", O_RDWR);
    if (udbfd < 0) {
        perror("open /dev/udmabuf (check if modprobe udmabuf is run)");
        return 1;
    }

    // 5. Define the udmabuf create structure
    struct udmabuf_create create;
    memset(&create, 0, sizeof(create));
    create.memfd = memfd;
    create.offset = 8192; // Start a bit into the memfd
    create.size = size - 8192; // Leave some offset for testing
    create.flags = UDMABUF_FLAGS_CLOEXEC;

    // 6. Create the DMA-BUF file descriptor
    dma_buf_fd = ioctl(udbfd, UDMABUF_CREATE, &create);
    if (dma_buf_fd < 0) {
        perror("ioctl UDMABUF_CREATE");
        close(udbfd);
        close(memfd);
        return 1;
    }

    // int kcmp_return = is_same_file(memfd, dma_buf_fd);
    // printf("kcmp return value (0 means same file): %d\n", kcmp_return);

    // if (verify_page_sharing(dma_buf_fd, memfd)) {
    //     printf("Verified: The two FDs share the same physical memory pages.\n");
    // } else {
    //     printf("Verification failed: The two FDs do not share the same physical memory pages.\n");
    // }

    int ret = find_memfd_for_udmabuf(dma_buf_fd, size);
    if (ret >= 0) {
        printf("Found matching memfd FD: %d\n", ret);
    } else {
        printf("No matching memfd FD found.\n");
    }

    ret = verify_dmabuf_memfd_link(dma_buf_fd, memfd);
    if (ret) {
        printf("Verified: DMA-BUF and memfd share the same memory via signature test.\n");
    } else {
        printf("Verification failed: DMA-BUF and memfd do not share the same memory.\n");
    }

    ret = find_udmabuf_offset(dma_buf_fd, memfd, size);
    if (ret >= 0) {
        printf("Found udmabuf offset within memfd: %ld bytes\n", ret);
    } else {
        printf("Failed to find udmabuf offset within memfd.\n");
    }

    if (fstat(dma_buf_fd, &st) == -1) {
		perror("fstat error");
		return -1;
	}
    // print every single field in st
    printf("DMA-BUF FD Stat:\n");
    printf("  st_dev: %lu\n", st.st_dev);
    printf("  st_ino: %lu\n", st.st_ino);
    printf("  st_mode: %o\n", st.st_mode);
    printf("  st_nlink: %lu\n", st.st_nlink);
    printf("  st_uid: %u\n", st.st_uid);
    printf("  st_gid: %u\n", st.st_gid);
    printf("  st_rdev: %lu\n", st.st_rdev);
    printf("  st_size: %ld\n", st.st_size);
    printf("  st_blksize: %ld\n", st.st_blksize);
    printf("  st_blocks: %ld\n", st.st_blocks);
    printf("  st_atime: %ld\n", st.st_atime);
    printf("  st_mtime: %ld\n", st.st_mtime);
    printf("  st_ctime: %ld\n", st.st_ctime);

    char path[256];
    read_fd_link(dma_buf_fd, path, 256);
    printf("DMA-BUF FD Link: %s\n", path);

    // close(udbfd); 
    // close(dma_buf_fd);

    // if (dma_buf_fd >= 0) {
    //     // Send dma_buf_fd to Mutter via Wayland protocol
    //     // Once sent, you can even close dma_buf_fd here if you 
    //     // don't need to modify the buffer anymore, as the 
    //     // receiving process (Mutter) will have its own reference.
    // }

    printf("Successfully created udmabuf FD: %d\n", dma_buf_fd);

    // Now dma_buf_fd can be passed to Mutter or PipeWire.
    while (true) {
	    count++;
    }

    close(dma_buf_fd);
    // close(udbfd);
    close(memfd);
    return 0;
}
