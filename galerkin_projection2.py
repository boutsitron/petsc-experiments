"""Experimenting with PETSc galerkin projection"""

import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD
from mpi4py import MPI
from numpy.testing import assert_array_almost_equal

from utilities import Print, create_petsc_matrix

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


galerkin_start = time.time()

# --------------------------------------------
# EXP: Galerkin projection of an mpi PETSc matrix A with an mpi PETSc matrix Phi
#  A' = Phi.T * A * Phi
# [k x k] <- [k x m] x [m x m] x [m x k]
# --------------------------------------------
m, k = 20000, 50
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.rand(m, m)
Phi_np = np.random.rand(m, k)

# Create A as an mpi matrix distributed on each process
A = create_petsc_matrix(A_np, sparse=True)

# Create Phi as an mpi matrix distributed on each process
Phi = create_petsc_matrix(Phi_np, sparse=True)

galerkin_setup = time.time()
galerkin_time = galerkin_setup - galerkin_start
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Setup A and Phi matrices: {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

# Perform the PtAP (Phi Transpose times A times Phi) operation.
# In mathematical terms, this operation is A' = Phi.T * A * Phi.
# A_prime will store the result of the operation.
AL = Phi.transposeMatMult(A)
# print_matrix_partitioning(AL, "AL")

# A_prime = AL.matMult(Phi)
A_prime = AL * Phi
# print_matrix_partitioning(A_prime, "A_prime")

galerkin_arom = time.time()
galerkin_time = galerkin_arom - galerkin_setup
galerkin_time_avg = COMM_WORLD.allreduce(galerkin_time, op=MPI.SUM) / nproc
Print(
    f"-Compute A' = Phi.T * A * Phi using traditional functions: {galerkin_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

# --------------------------------------------
# TEST: Galerking projection of numpy matrices A_np and Phi_np
# --------------------------------------------
galerkin_start = time.time()

A_prime = A.ptap(Phi)
# print_matrix_partitioning(A_prime, "A_prime")

galerkin_arom = time.time()
Print(
    f"-Compute A' = Phi.T * A * Phi using ptap(): {galerkin_arom - galerkin_start: 2.2f} s",
    Fore.MAGENTA,
)

# --------------------------------------------
# TEST: Galerking projection of numpy matrices A_np and Phi_np
# --------------------------------------------
Aprime_np = Phi_np.T @ A_np @ Phi_np
Print(f"MATRIX Aprime_np [{Aprime_np.shape[0]}x{Aprime_np.shape[1]}]")
Print(f"{Aprime_np}")

# Get the local values from C
local_rows_start, local_rows_end = A_prime.getOwnershipRange()
Aprime = A_prime.getValues(range(local_rows_start, local_rows_end), range(k))

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(
    Aprime, Aprime_np[local_rows_start:local_rows_end, :], decimal=5
)
