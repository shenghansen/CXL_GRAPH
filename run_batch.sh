echo "Running batch script"
sudo accel-config config-user-default 
output_log="output.log.56"
if [ -f "$output_log" ]; then
    echo "Removing existing output log file: $output_log"
    rm "$output_log"
fi


# echo "#######################################################################" >> $output_log
#  ./clear_shm.sh 
# SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 "
# cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
# cmake --build build -- -v
# ./run.script >> $output_log 2>&1
#  ./clear_shm.sh 

# echo "#######################################################################" >> $output_log
#  ./clear_shm.sh 
# SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL "
# cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
# cmake --build build -- -v
# ./run.script >> $output_log 2>&1

# echo "#######################################################################" >> $output_log
#  ./clear_shm.sh 
# SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL -DPREFETCH"
# cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
# cmake --build build -- -v
# ./run.script >> $output_log 2>&1

# echo "#######################################################################" >> $output_log
#  ./clear_shm.sh 
# SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL -DPREFETCH -DSPARSE_MODE_UNIDIRECTIONAL -DDENSE_MODE_UNIDIRECTIONAL -DGET"
# cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
# cmake --build build -- -v
# ./run.script >> $output_log 2>&1

# echo "#######################################################################" >> $output_log
#  ./clear_shm.sh 
# SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL -DPREFETCH -DGLOBAL_STEALING_DENSE"
# cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
# cmake --build build -- -v
# ./run.script >> $output_log 2>&1

echo "#######################################################################" >> $output_log
 ./clear_shm.sh 
SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL -DPREFETCH -DSPARSE_MODE_UNIDIRECTIONAL -DDENSE_MODE_UNIDIRECTIONAL -DGET -DGLOBAL_STEALING_DENSE"
cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
cmake --build build -- -v
./run.script >> $output_log 2>&1 



# SHELL_DEFINES_TO_ADD="-DEXEC_TIMES=10 -DCXL -DPREFETCH"
# cmake -S . -B build -DDEFINES_TO_ADD="${SHELL_DEFINES_TO_ADD}"
# cmake --build build -- -v
# OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_PLACES=cores sudo  mpirun -np 4 --allow-run-as-root --map-by socket  ./build/bfs ../data/enwiki-2013in 4206785 1