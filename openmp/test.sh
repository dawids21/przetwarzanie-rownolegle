#!/bin/bash
n_start=4
n_current=$n_start
n_end=14
step=1

if [ -f result.txt ]; then
    rm result.txt
fi

while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        ./hamilton s $n_current 0.6 1 >>result.txt
    done
    let n_current=n_current+step
done

level=1
let n_current=n_start+level-1
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        ./hamilton p $n_current 0.6 $level >>result.txt
    done
    let n_current=n_current+step
done

level=2
let n_current=n_start+level-1
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        ./hamilton p $n_current 0.6 $level >>result.txt
    done
    let n_current=n_current+step
done

level=3
let n_current=n_start+level-1
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        ./hamilton p $n_current 0.6 $level >>result.txt
    done
    let n_current=n_current+step
done

level=7
let n_current=n_start+level-1
while [ $n_current -le $n_end ]; do
    for i in {1..10}; do
        ./hamilton p $n_current 0.6 $level >>result.txt
    done
    let n_current=n_current+step
done
