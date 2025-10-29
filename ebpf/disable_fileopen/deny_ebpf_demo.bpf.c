#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>
#include <linux/errno.h>

SEC("lsm/file_open")
int BPF_PROG(deny_ebpf_demo, struct file *file)
{
    if (!file)
        return 0;

    struct dentry *d = BPF_CORE_READ(file, f_path.dentry);
    if (!d)
        return 0;

    struct qstr d_name = BPF_CORE_READ(d, d_name);
    char fname[64] = {};
    if (bpf_core_read_str(fname, sizeof(fname), d_name.name) < 0)
        return 0;

    if (bpf_strncmp(fname, 64, "ebpfdemo.txt") != 0) {
        return 0;
    } else {
        bpf_printk("deny_ebpf_demo: denied open of %s\n", fname);
        return -EACCES;
    }
}

char LICENSE[] SEC("license") = "GPL";
