#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <unistd.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))

int main() {
    int length, i = 0;
    int fd;
    int wd;
    char buffer[BUF_LEN];

    // print pid
    printf("Process PID: %d\n", getpid());

    // 1. Initialize inotify
    fd = inotify_init();
    if (fd < 0) {
        perror("inotify_init");
    }

    // 2. Add a watch for the current directory ("./")
    // We watch for Create, Delete, and Modify events
    wd = inotify_add_watch(fd, "./", IN_CREATE | IN_DELETE | IN_MODIFY);

    printf("Watching directory for changes...\n");

    while (1) {
        i = 0;
        // 3. Read events (this blocks until an event occurs)
        length = read(fd, buffer, BUF_LEN);

        if (length < 0) {
            perror("read");
        }

        // 4. Process the events in the buffer
        while (i < length) {
            struct inotify_event *event = (struct inotify_event *) &buffer[i];
            if (event->len) {
                if (event->mask & IN_CREATE) {
                    printf("The file %s was created.\n", event->name);
                } else if (event->mask & IN_DELETE) {
                    printf("The file %s was deleted.\n", event->name);
                } else if (event->mask & IN_MODIFY) {
                    printf("The file %s was modified.\n", event->name);
                }
            }
            // Move to the next event in the buffer
            i += EVENT_SIZE + event->len;
        }
    }

    // 5. Clean up
    inotify_rm_watch(fd, wd);
    close(fd);

    return 0;
}