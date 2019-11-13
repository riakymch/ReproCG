#!/bin/sh
#

#PBS -q hasw
#PBS -l nodes=2:ppn=16
#PBS -l walltime=04:00:00
#

# Header Information

# Define local ditrectory
cd /home/mvaya/CG_MPI_Sparse_Roman/MPI

export PATH=/state/partition1/soft/gnu/gcc-5.3.0/bin:$PATH
export LD_LIBRARY_PATH=/state/partition1/soft/gnu/gcc-5.3.0/lib64:/state/partition1/soft/intel/composer_xe_2011_sp1/mkl/lib/intel64/:/state/partition1/soft/intel/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64_lin/:$LD_LIBRARY_PATH

ldd ./CG_MPI
procs=4
mat="../Matrices/A050.rsa"
mpirun -np $procs ./CG_MPI $mat

