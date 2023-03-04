#!/usr/bin/bash

for n_proc in 1 2 3 4 5 6 7 8
do
    # echo $n_proc

    cat data_write_test.cpp | head -n $n_proc
    echo "------------------------------------------"

    # g++ -o a.out data_write_test.cpp
    # ./a.out >> log.txt
done



