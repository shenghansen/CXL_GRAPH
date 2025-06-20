#!/bin/bash
echo "Running batch script"
sudo accel-config config-user-default 
output_log="output.log.64"
if [ -f "$output_log" ]; then
    echo "Removing existing output log file: $output_log"
    rm "$output_log"
fi

echo "#######################################################################" >> $output_log
 ./clear_shm.sh 
SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL -DPREFETCH -DSPARSE_MODE_UNIDIRECTIONAL -DDENSE_MODE_UNIDIRECTIONAL -DGET -DGLOBAL_STEALING_DENSE"
cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
cmake --build build -- -v


commands=( "./build/pagerank" "./build/bfs"  "./build/sssp" "./build/bc")
# commands=(  "./build/bfs" )
args=( "../data/twitter-2010Bin 41652230 1" "../data/enwiki-2013in 4206785 1" "../data/uk-2007Bin 105896555 1"  "../data/rMat24Bin 16800000 1" "../data/rMat27Bin 134000000 1")
#  args=(   "../data/rMat24Bin 16800000 1" )
#"./build/pagerank" "./build/bfs"  "./build/sssp"
#"./build/cc"
num_commands=${#commands[@]}
num_args=${#args[@]}


for ((i=0; i<$num_commands; i++)); do
  for ((j=0; j<$num_args; j++)); do
    command=${commands[$i]}
    param=${args[$j]}
    echo "run: OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo  mpirun -np 1 --allow-run-as-root --map-by socket  $command $param"
    OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo mpirun --allow-run-as-root -np 1 --map-by socket $command $param >> $output_log 2>&1 
  done
done

for ((i=0; i<$num_commands; i++)); do
  for ((j=0; j<$num_args; j++)); do
    command=${commands[$i]}
    param=${args[$j]}
    echo "run: OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo  mpirun -np 2 --allow-run-as-root --map-by socket  $command $param"
    OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo mpirun --allow-run-as-root -np 2 --map-by socket $command $param >> $output_log 2>&1
  done
done

for ((i=0; i<$num_commands; i++)); do
  for ((j=0; j<$num_args; j++)); do
    command=${commands[$i]}
    param=${args[$j]}
    echo "run: OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo  mpirun -np 4 --allow-run-as-root --map-by socket  $command $param"
    OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo mpirun --allow-run-as-root -np 4 --map-by socket $command $param >> $output_log 2>&1
  done
done

