TARGETS = 0_vmadot 1_vmadot_vd 2_vmadot_tiled_vd

all: $(TARGETS)

0_vmadot: 0_vmadot.c
	gcc 0_vmadot.c -O3 -march=rv64gcv -o $@

1_vmadot_vd: 1_vmadot_vd.c
	gcc 1_vmadot_vd.c -O3 -march=rv64gcv -o $@

2_vmadot_tiled_vd: 2_vmadot_tiled_vd.c ime_gemm.S
	gcc 2_vmadot_tiled_vd.c ime_gemm.S -O3 -march=rv64gcv -o $@

ifeq ($(shell uname -m), riscv64)
run: $(TARGETS)
	@echo "=========================================="
	@echo "1. Running Baseline (0_vmadot.c)"
	@echo "=========================================="
	taskset -c 0 ./0_vmadot
	@echo ""
	@echo "=========================================="
	@echo "2. Running Virtual Dimension (1_vmadot_vd.c)"
	@echo "=========================================="
	taskset -c 0 ./1_vmadot_vd
	@echo ""
	@echo "=========================================="
	@echo "3. Running Tiled VD (2_vmadot_tiled_vd.c)"
	@echo "=========================================="
	taskset -c 0 ./2_vmadot_tiled_vd
	@echo ""
else
run:
	@echo "=========================================="
	@echo "🚀 Transferring source codes to Banana Pi (bpi) board..."
	@echo "=========================================="
	ssh bpi 'mkdir -p ~/ime_workspace/2mm'
	scp $(addsuffix .c, $(TARGETS)) ime_gemm.S Makefile bpi:~/ime_workspace/2mm/
	@echo "=========================================="
	@echo "⚙️  Starting native compilation and execution on the board..."
	@echo "=========================================="
	ssh bpi 'cd ~/ime_workspace/2mm && make clean && make run'
	@echo "=========================================="
	@echo "✅ Remote execution completed"
	@echo "=========================================="
endif

clean:
	rm -rf $(TARGETS) *.s

s:
	@echo "=========================================="
	@echo "🛠️  Generating assembly on the board..."
	@echo "=========================================="
	ssh bpi 'mkdir -p ~/ime_workspace/2mm'
	scp $(addsuffix .c, $(TARGETS)) ime_gemm.S Makefile bpi:~/ime_workspace/2mm/
	ssh bpi 'cd ~/ime_workspace/2mm && gcc 0_vmadot.c -S -O3 -fno-unroll-loops -march=rv64gcv -o 0_vmadot.s'
	ssh bpi 'cd ~/ime_workspace/2mm && gcc 1_vmadot_vd.c -S -O3 -fno-unroll-loops -march=rv64gcv -o 1_vmadot_vd.s'
	ssh bpi 'cd ~/ime_workspace/2mm && gcc 2_vmadot_tiled_vd.c -S -O3 -fno-unroll-loops -march=rv64gcv -o 2_vmadot_tiled_vd.s'
	scp bpi:~/ime_workspace/2mm/*.s ./
	@echo "=========================================="
	@echo "✅ Assembly fetched successfully"
	@echo "=========================================="
