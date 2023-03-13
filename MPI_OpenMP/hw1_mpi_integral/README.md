
### Usage


1) add mpi module
   
```sh
module add centos/8/mpi/hpcx-v2.7.0
```

2) compile .cpp file
   
```sh
mpiCC main.cpp
```

3) run single experiment using slurm (as example)
   
```sh
sbatch ./run.sh
```

4) run multiple experiments using slurm
   
```sh
./run_multiple_experiments.sh
```

5) draw acceleration graph

```sh
python3 draw_acceleration_graph.py
```
