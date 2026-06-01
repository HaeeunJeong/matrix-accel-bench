TARGET=pure_c_vd
CFLAGS=-O2 -mcpu=spacemit-x60 -mabi=lp64d -Wall

# Check architecture
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),riscv64)
    CC=gcc
    EXEC=taskset -c 0 ./$(TARGET)
else
    CC=riscv64-unknown-linux-gnu-gcc
endif

$(TARGET) : $(TARGET).c
	$(CC) $(TARGET).c $(CFLAGS) -o $(TARGET)

.PHONY: run s clean

run:
	@if [ "$(UNAME_M)" = "riscv64" ]; then \
		$(CC) $(TARGET).c $(CFLAGS) -o $(TARGET); \
		taskset -c 0 ./$(TARGET); \
	else \
		echo "Deploying to Banana Pi for execution..."; \
		ssh bpi 'mkdir -p ~/ime_workspace/c_mm'; \
		scp $(TARGET).c Makefile bpi:~/ime_workspace/c_mm/; \
		ssh bpi 'cd ~/ime_workspace/c_mm && gcc $(TARGET).c $(CFLAGS) -o $(TARGET) && taskset -c 0 ./$(TARGET)'; \
	fi

s:
	@if [ "$(UNAME_M)" = "riscv64" ]; then \
		$(CC) $(TARGET).c $(CFLAGS) -S -o $(TARGET).s; \
	else \
		echo "Generating assembly on Banana Pi and fetching..."; \
		ssh bpi 'mkdir -p ~/ime_workspace/c_mm'; \
		scp $(TARGET).c Makefile bpi:~/ime_workspace/c_mm/; \
		ssh bpi 'cd ~/ime_workspace/c_mm && gcc $(TARGET).c $(CFLAGS) -S -o $(TARGET).s'; \
		scp bpi:~/ime_workspace/c_mm/$(TARGET).s ./; \
	fi

clean:
	rm -f $(TARGET) $(TARGET).s
