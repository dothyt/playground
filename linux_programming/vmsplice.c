#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/uio.h>

int main() {
    int pipefd[2];
    void *buffer;

    // Get the system's memory page size (usually 4096 bytes)
    size_t page_size = getpagesize();

    // 1. Create the pipe that will act as our kernel-space conduit
    if (pipe(pipefd) < 0) {
        perror("pipe creation failed");
        return 1;
    }

    // 2. Allocate strictly page-aligned memory for true zero-copy
    if (posix_memalign(&buffer, page_size, page_size) != 0) {
        perror("Memory allocation failed");
        return 1;
    }

    // Write our message into the buffer
    const char *msg = "Hello from the vmsplice zero-copy world!\n";
    strcpy((char *)buffer, msg);

    // 3. Define the memory region we want to vmsplice
    struct iovec iov;
    iov.iov_base = buffer;
    iov.iov_len = page_size; // We must gift the entire page

    // 4. vmsplice: Map the physical page directly into the pipe's write end
    // WARNING: Because we use SPLICE_F_GIFT, we promise not to modify 'buffer' after this!
    ssize_t bytes_vmspliced = vmsplice(pipefd[1], &iov, 1, SPLICE_F_GIFT);
    if (bytes_vmspliced < 0) {
        perror("vmsplice failed");
        return 1;
    }

    // 5. splice: Move the data from the pipe's read end directly to Standard Output (fd 1)
    // We only splice the length of our actual string so we don't print trailing garbage
    ssize_t bytes_spliced = splice(pipefd[0], NULL, STDOUT_FILENO, NULL, strlen(msg), 0);
    if (bytes_spliced < 0) {
        perror("splice failed");
        return 1;
    }

    // Clean up
    free(buffer);
    close(pipefd[0]);
    close(pipefd[1]);

    return 0;
}
