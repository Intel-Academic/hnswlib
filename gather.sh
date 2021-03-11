#!/usr/bin/env bash
set -euo pipefail

m=48
ef_construction=200
ef="1,2,5,10,15,20,25,30,40,50,60,70,80,90,110,100,120,130,140"

subset=1,10,100,200,500,1000

#notes="HT enabled,NAND SSD,DRAM=96GB"
notes=$1

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/intel/oneapi/vtune/latest/lib64

avail_mem=`awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo`

for sub in 1 10 100 200 500 1000
do
    if [[ $avail_mem -le 96 ]] && [[ ${sub} -eq 1000  ]];
    then
        echo "Reducing max threads on 1B dataset due to memory constraint"
        threads="1,2,4,8,16,20,24,28" # More threads will crash due to OOM
    else
        threads="1,2,4,8,16,20,24,28,32,40,48,56"
    fi
    ~/src/hnswlib/build/main -m ${m} --subset ${sub} --ef-construction ${ef_construction} \
        --ef ${ef} --n_queries 10000 -k 1 --repeat 12 --qsize 10000 --permute --algorithm hnsw \
        --threads $threads \
        --notes "${notes}"
done
