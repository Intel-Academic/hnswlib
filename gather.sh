#!/usr/bin/env bash
set -euo pipefail

m=48
ef_construction=200
ef="10,25,50,75,80,100,120,140,160"
threads="1,2,4,8,16,32,48,56,64,72,96"

subset=1,10,100,200,500,1000

notes="HT enabled"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/intel/oneapi/vtune/latest/lib64

for sub in 1 10 100 200 500 1000
do
           ~/src/hnswlib/build/main -m ${m} --subset ${sub} --ef-construction ${ef_construction} \
               --ef ${ef} --n_queries 10000 -k 1 --repeat 5 --qsize 10000 --permute --algorithm hnsw \
               --threads $threads \\
               --notes "${notes}" \\
done
