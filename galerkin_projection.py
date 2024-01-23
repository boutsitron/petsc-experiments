"""Experimenting with PETSc mat-mat multiplication"""

import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD
from mpi4py import MPI
from numpy.testing import assert_array_almost_equal

from utilities import (
    Print,
    concatenate_local_to_global_matrix,
    convert_global_matrix_to_seq,
    convert_seq_matrix_to_global,
    create_petsc_matrix,
    create_petsc_matrix_seq,
    get_local_submatrix,
)

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank

galerkin_start = time.time()

# --------------------------------------------
# EXP: Galerkin projection of an mpi PETSc matrix A with an mpi PETSc matrix Phi
#  A' = Phi.T * A * Phi
# [k x k] <- [k x m] x [m x m] x [m x k]
# --------------------------------------------

m, k = 5000, 30
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.rand(m, m)
Phi_np = np.random.rand(m, k)

# Create A as an mpi matrix distributed on each process
A = create_petsc_matrix(A_np, sparse=True)
# print_matrix_partitioning(A, "A")
# Create Phi as an mpi matrix distributed on each process
Phi = create_petsc_matrix(Phi_np, sparse=True)
# print_matrix_partitioning(Phi, "Phi")

galerkin_setup = time.time()
galerkin_time = galerkin_setup - galerkin_start
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Setup A and Phi matrices: {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

# --------------------------------------------
# TEST: Using custom parallel/sequential functions to compute A' = Phi.T * A * Phi
# --------------------------------------------

# Getting the correct local submatrix to be multiplied by Phi
A_local = get_local_submatrix(A)
# print_matrix_partitioning(A_local, "A_local")

# Get a Phi matrix that is sequential from the distributed Phi
Phi_seq = convert_global_matrix_to_seq(Phi)
# print_matrix_partitioning(Phi_seq, "Phi_seq")

# Step 1: Compute Aphi = A * Phi
APhi_local = create_petsc_matrix_seq(np.zeros((m, k)))
APhi_local = A_local.matMult(Phi_seq)
# print_matrix_partitioning(APhi_local, "APhi_local")

# Creating the global Aphi matrix
APhi = concatenate_local_to_global_matrix(APhi_local) if nproc > 1 else APhi_local
APhi_seq = convert_global_matrix_to_seq(APhi)


# Step 2: Compute A' = Phi.T * APhi
A_prime_seq = create_petsc_matrix_seq(np.zeros((k, k)))
A_prime_seq = Phi_seq.transposeMatMult(APhi_seq)
# print_matrix_partitioning(A_prime_seq, "A_prime_seq")

A_prime = convert_seq_matrix_to_global(A_prime_seq)
# print_matrix_partitioning(A_prime, "A_prime")

galerkin_seq = time.time()
galerkin_time = galerkin_seq - galerkin_setup
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Compute A' = Phi.T * A * Phi using Thanos's functions: {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

# --------------------------------------------
# TEST: Using PETSc functions to compute A' = Phi.T * A * Phi
# --------------------------------------------

# First compute AL = Phi.T * A, then compute A' = AL * Phi
AL = Phi.transposeMatMult(A)
# print_matrix_partitioning(AL, "AL")

# A_prime = AL.matMult(Phi)
A_prime = AL * Phi
# print_matrix_partitioning(A_prime, "A_prime")

galerkin_parallel1 = time.time()
galerkin_time = galerkin_parallel1 - galerkin_seq
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Compute A' = Phi.T * A * Phi using PETSc (left first): {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

# --------------------------------------------
# TEST: Using PETSc functions to compute A' = Phi.T * A * Phi
# --------------------------------------------

# First compute AR = A * Phi, then compute A' = Phi.T * AR
AR = A * Phi
# print_matrix_partitioning(AR, "AR")

A_prime = Phi.transposeMatMult(AR)
# print_matrix_partitioning(A_prime, "A_prime")

galerkin_parallel2 = time.time()
galerkin_time = galerkin_parallel2 - galerkin_seq
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Compute A' = Phi.T * A * Phi using PETSc (right first): {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

A_prime_seq = convert_global_matrix_to_seq(A_prime)
# --------------------------------------------
# TEST: Using ptap() to compute A' = Phi.T * A * Phi
# --------------------------------------------

A_prime = A.ptap(Phi)
# print_matrix_partitioning(A_prime, "A_prime")

galerkin_ptap = time.time()
galerkin_time = galerkin_ptap - galerkin_parallel2
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Compute A' = Phi.T * A * Phi using ptap(): {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)


# --------------------------------------------
# TEST: Galerking projection of numpy matrices A_np and Phi_np
# --------------------------------------------
Aprime_np = Phi_np.T @ A_np @ Phi_np
Print(f"MATRIX Aprime_np [{Aprime_np.shape[0]}x{Aprime_np.shape[1]}]")
# Print(f"{Aprime_np}")

# Get the local values from C
local_rows_start, local_rows_end = A_prime.getOwnershipRange()
Aprime = A_prime.getValues(range(local_rows_start, local_rows_end), range(k))

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(
    Aprime, Aprime_np[local_rows_start:local_rows_end, :], decimal=5
)
