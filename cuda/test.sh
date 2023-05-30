#!/bin/bash
n_start=20
n_current=$n_start
n_end=1000
step=20

if [ -f result.txt ]; then
    rm result.txt
fi

while [ $n_current -le $n_end ]; do
    for i in {1..5}; do
        ./mandelbrot c $n_current 1000 >>result.txt
    done
    let n_current=n_current+step
done