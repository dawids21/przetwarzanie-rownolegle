#!/bin/bash
n_start=5
n_current=$n_start
n_end=14
step=1

if [ -f result.txt ]; then
    rm result.txt
fi

threads=5
n_current=$n_start
echo $threads >>result.txt
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        OMP_NUM_THREADS=$threads ./hamilton p $n_current 0.4 2 >>result.txt
    done
    let n_current=n_current+step
done

threads=10
n_current=$n_start
echo $threads >>result.txt
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        OMP_NUM_THREADS=$threads ./hamilton p $n_current 0.4 2 >>result.txt
    done
    let n_current=n_current+step
done

threads=20
n_current=$n_start
echo $threads >>result.txt
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        OMP_NUM_THREADS=$threads ./hamilton p $n_current 0.4 2 >>result.txt
    done
    let n_current=n_current+step
done

threads=40
n_current=$n_start
echo $threads >>result.txt
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        OMP_NUM_THREADS=$threads ./hamilton p $n_current 0.4 2 >>result.txt
    done
    let n_current=n_current+step
done
