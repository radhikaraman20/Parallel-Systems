#Group info:
# psomash Prakruthi Somashekarappa
# rbraman Radhika B Raman
# srames22 Srivatsan Ramesh
p2_mpi: p2_mpi.c p2_func.c
	mpicc -lm -O3 -o p2_mpi p2_mpi.c p2_func.c
