### profile open syscall latency
```
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @start[tid] = nsecs(); }
                  tracepoint:syscalls:sys_exit_openat  { @latency = hist(nsecs() - @start[tid]); delete(@start[tid]); }'
```

### genereate vmlinux.h
```
sudo bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
```

### disable a file being opened
```
# you will need to regenerate the vmlinux.h from the command above
make
sudo ./demo_loader
# observe the file cannot be opened anymore
cat ebpfdemo.txt
```

### bpf_printk output location
```
sudo cat /sys/kernel/debug/tracing/trace_pipe
```

### show all running ebpf program
```
sudo bpftool show prog
```
