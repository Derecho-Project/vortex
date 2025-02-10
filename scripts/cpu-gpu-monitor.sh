#!/bin/bash

mkdir data
cd data

# apt-get install moreutils
# needed for ts

while true
do
    top -n 1 -o +%CPU -b | grep cascade_server | sed -e 's/\s\+/,/g' | ts %s >> cpu_utilization.dat
    { date +%s | sed -z 's/\n/, /g'; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader; } >> gpu_utilization.dat
    nvidia-smi --query-compute-apps=process_name,pid,used_memory --format=csv,noheader | ts %s"," >> gpu_by_process.dat
    sleep 3
done
