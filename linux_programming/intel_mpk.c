#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>

int shared_pkey;
char *shared_buffer;

// Thread 1: The "Worker" - needs read/write access
void* worker_thread(void* arg) {
    // Modify this thread's LOCAL PKRU register to allow full access to the key
    pkey_set(shared_pkey, 0); // 0 means clear AD and WD bits

    // This will succeed
    shared_buffer[0] = 'W';
    printf("Worker thread successfully wrote to the buffer.\n");

    return NULL;
}

// Thread 2: The "Untrusted/Monitor" - should not touch the memory
void* untrusted_thread(void* arg) {
    // Modify this thread's LOCAL PKRU register to disable access to the key
    pkey_set(shared_pkey, PKEY_DISABLE_ACCESS); // Sets the AD bit

    printf("Untrusted thread attempting to read buffer...\n");

    // THIS WILL TRIGGER A SIGSEGV (Segmentation Fault) in this thread only!
    char c = shared_buffer[0];

    return NULL;
}

int main() {
    // 1. Allocate a page of memory
    int page_size = getpagesize();
    shared_buffer = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    // 2. Allocate a hardware protection key from the kernel
    shared_pkey = pkey_alloc(0, 0);

    // 3. Tag the shared memory page with our new key in the Page Tables
    // (This tag is now visible to ALL threads)
    pkey_mprotect(shared_buffer, page_size, PROT_READ | PROT_WRITE, shared_pkey);

    // 4. Spawn threads
    pthread_t t1, t2;
    pthread_create(&t1, NULL, worker_thread, NULL);
    pthread_create(&t2, NULL, untrusted_thread, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
