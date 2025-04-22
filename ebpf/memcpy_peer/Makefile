# Makefile for libbpf application using local submodule and CO-RE

# Application name
APP := cudamemcpypeer_track

# Source files
BPF_SRC := $(wildcard *.bpf.c)
USER_SRC := $(wildcard *.c)
USER_SRC := $(filter-out $(BPF_SRC), $(USER_SRC)) # Exclude BPF source

# Object files
BPF_OBJ := $(BPF_SRC:.bpf.c=.bpf.o)
USER_OBJ := $(USER_SRC:.c=.o)

# Generated files
SKELETON := $(BPF_SRC:.bpf.c=.skel.h)
VMLINUX_H := vmlinux.h

# Tools
CLANG ?= clang
LLVM_STRIP ?= llvm-strip
GCC ?= gcc
BPFTOOL ?= /usr/lib/linux-tools/$(shell uname -r)/bpftool
MAKE ?= make

# libbpf submodule location
LIBBPF_SRC := $(abspath ../../../libbpf/src)
LIBBPF_OBJ := $(LIBBPF_SRC)/build/libbpf.a # Assuming static build in build/ subdir

# Build variables
ARCH := $(shell uname -m | sed 's/x86_64/x86/')

# Flags for userspace code (include headers from libbpf submodule)
CFLAGS := -g -O2 -Wall -I$(LIBBPF_SRC) -I.
# Linker flags (link against libbpf.a from submodule and dependencies)
LDFLAGS := -L$(LIBBPF_SRC)/build # Or -L$(LIBBPF_SRC) if built directly in src/
LDLIBS := -lbpf -lelf -lz # Link order can matter: app obj, static libs, shared libs

# Flags for BPF code (include headers from libbpf submodule)
BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH)
BPF_CFLAGS += -I$(LIBBPF_SRC)
BPF_CFLAGS += -I$(abspath .) # For vmlinux.h

# Default target
all: $(APP)

# Build static libbpf library from submodule if not already built
# Note: Assumes libbpf's Makefile uses OBJDIR=build for static build output
# Adjust LIBBPF_OBJ path if libbpf builds libbpf.a directly in src/
$(LIBBPF_OBJ):
	$(MAKE) -C $(LIBBPF_SRC) OBJDIR=build STATIC_LIB=1

# Link the final application executable
$(APP): $(USER_OBJ) $(SKELETON) $(LIBBPF_OBJ)
	$(GCC) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

# Compile user space source files
%.o: %.c $(SKELETON) $(VMLINUX_H)
	$(GCC) $(CFLAGS) -c $< -o $@

# Generate BPF skeleton header from BPF object file
$(SKELETON): $(BPF_OBJ) $(BPFTOOL)
	$(BPFTOOL) gen skeleton $< > $@

# Compile BPF C code to BPF object file
$(BPF_OBJ): $(BPF_SRC) $(VMLINUX_H)
	$(CLANG) $(BPF_CFLAGS) -c $(filter %.bpf.c,$^) -o $@

# Generate vmlinux.h from kernel BTF info (for CO-RE)
$(VMLINUX_H):
	@echo "--- Checking for kernel BTF info ---"
	@if [ -f /sys/kernel/btf/vmlinux ]; then \
		echo "Found /sys/kernel/btf/vmlinux. Attempting to generate $(VMLINUX_H)..."; \
		$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@.tmp; \
		if [ $$? -eq 0 ]; then \
			echo "$(VMLINUX_H) generated successfully from BTF."; \
			mv $@.tmp $@; \
		else \
			echo "WARN: 'bpftool btf dump' failed. $(VMLINUX_H) could not be generated from BTF."; \
			rm -f $@.tmp; \
			echo "WARN: Creating empty $(VMLINUX_H) as fallback."; touch $@; \
		fi; \
	else \
		echo "WARN: Cannot find /sys/kernel/btf/vmlinux. Kernel may lack BTF info."; \
		echo "WARN: Creating empty $(VMLINUX_H) as fallback."; touch $@; \
	fi
	@echo "--- Finished checking for kernel BTF info ---"

# Clean target
clean:
	rm -f $(APP) $(BPF_OBJ) $(USER_OBJ) $(SKELETON) $(VMLINUX_H) $(VMLINUX_H).tmp
	# Optionally clean libbpf submodule build artifacts
	# $(MAKE) -C $(LIBBPF_SRC) clean OBJDIR=build

# Phony targets
.PHONY: all clean $(LIBBPF_OBJ)