#include <stdio.h>
#include <stdlib.h>
#include <systemd/sd-bus.h>
#include <criu/criu-plugin.h>
#include <sys/socket.h>
#include <sys/un.h>

#define DBUS_PATH "/run/user/1000/bus"
#define META_FILE "dbus_meta.img"

struct dbus_meta {
    int id;
    char well_known_name[256];
};

int cr_plugin_dump_unix_sk(int sk, int id) {
    sd_bus *bus = NULL;
    char **names = NULL;
    pid_t pid = criu_get_item_pid();
    int r;

    r = sd_bus_open_user(&bus);
    if (r < 0) return 0;

    r = sd_bus_list_names(bus, &names, NULL);
    if (r >= 0) {
        for (char **name = names; *name; name++) {
            uint32_t owner_pid;
            if (sd_bus_get_name_creds(bus, *name, SD_BUS_CREDS_PID, NULL, &owner_pid) >= 0) {
                if (owner_pid == (uint32_t)pid && (*name)[0] != ':') {
                    struct dbus_meta meta = { .id = id };
                    strncpy(meta.well_known_name, *name, 255);
                    
                    int fd = open(META_FILE, O_WRONLY | O_CREAT | O_APPEND, 0644);
                    write(fd, &meta, sizeof(meta));
                    close(fd);
                    break; 
                }
            }
        }
        strv_free(names);
    }
    sd_bus_unref(bus);
    return 0;
}

/* --- RESTORE PHASE: Re-authenticating --- */
int cr_plugin_restore_unix_sk(int id) {
    int meta_fd = open(META_FILE, O_RDONLY);
    if (meta_fd < 0) return -1;

    struct dbus_meta meta;
    int found = 0;
    while (read(meta_fd, &meta, sizeof(meta)) > 0) {
        if (meta.id == id) { found = 1; break; }
    }
    close(meta_fd);
    if (!found) return -1;

    int sk = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    strncpy(addr.sun_path, DBUS_PATH, sizeof(addr.sun_path)-1);
    connect(sk, (struct sockaddr *)&addr, sizeof(addr));

    sd_bus *bus = NULL;
    sd_bus_new(&bus);
    sd_bus_set_fd(bus, sk, sk);
    sd_bus_start(bus);

    sd_bus_request_name(bus, meta.well_known_name, 0);

    // 4. Detach sd-bus so it doesn't close the FD when we unref
    // We want to return a "raw" authenticated FD to CRIU
    int final_fd = dup(sk); 
    sd_bus_flush_close_unref(bus); 
    close(sk);

    return final_fd; 
}
