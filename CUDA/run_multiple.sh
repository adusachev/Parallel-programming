#!/bin/bash

for BlockSize in 4 8 16 32
    do
        nvcc MatrixMul.cu
        ./a.out $BlockSize
    done

