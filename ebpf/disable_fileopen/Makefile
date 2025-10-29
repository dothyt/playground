BPF_CLANG ?= clang
BPF_CFLAGS ?= -O2 -g -target bpf

CC ?= gcc

LIBBPF_CFLAGS := $(shell pkg-config --cflags libbpf)
LIBBPF_LIBS   := $(shell pkg-config --libs libbpf)

all: deny_ebpf_demo.bpf.o deny_ebpf_demo.skel.h demo_loader

deny_ebpf_demo.bpf.o: deny_ebpf_demo.bpf.c
	$(BPF_CLANG) $(BPF_CFLAGS) -I. -c $< -o $@

deny_ebpf_demo.skel.h: deny_ebpf_demo.bpf.o
	bpftool gen skeleton $< > $@

demo_loader: demo_loader.c
	$(CC) -O2 demo_loader.c -o demo_loader $(LIBBPF_CFLAGS) $(LIBBPF_LIBS)

clean:
	rm -f *.o *.skel.h demo_loader
