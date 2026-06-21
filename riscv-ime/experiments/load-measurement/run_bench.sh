#!/bin/bash

MAX_RETRIES=5
SLEEP_TIME=5

retry_cmd() {
    local n=1
    while true; do
        if "$@"; then
            return 0
        else
            if [[ $n -lt $MAX_RETRIES ]]; then
                echo "Command failed. Retrying ($n/$MAX_RETRIES) in $SLEEP_TIME seconds..."
                sleep $SLEEP_TIME
                ((n++))
            else
                echo "Command failed after $MAX_RETRIES attempts: $@"
                return 1
            fi
        fi
    done
}

echo "Copying load_bench.c to bpi..."
if ! retry_cmd scp load_bench.c bpi:~/load_bench.c; then
    exit 1
fi

echo "Compiling and running on bpi..."
if ! retry_cmd ssh bpi "
    sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 2>/dev/null || true;
    gcc -O3 -march=rv64gcv load_bench.c -o load_bench -lm && taskset -c 0 ./load_bench
"; then
    exit 1
fi
