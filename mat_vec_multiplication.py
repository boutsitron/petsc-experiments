"""Experimenting with PETSc mat-vec multiplication"""

import time

import numpy as np
from colorama import Fore
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI
from numpy.testing import assert_array_almost_equal

from utilities import (
    Print,
    create_petsc_matrix,
    create_petsc_vector,
    create_petsc_vector_seq,
    get_local_submatrix,
    get_local_subvector,
    print_matrix_partitioning,
    print_vector_partitioning,
)

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def convert_global_vector_to_seq(v_global):
    """
    Fetch the complete global vector `v_global` to a local sequential vector on all ranks.

    Args:
        v_global (PETSc Vec): Global MPI PETSc vector.

    Returns:
        PETSc Vec: Local sequential vector having the complete values of `v_global`.
    """
    # Create a local sequential vector of the same size as the global vector.
    global_size = v_global.getSize()
    v_seq = PETSc.Vec().createSeq(global_size, comm=COMM_SELF)

    # Create index sets for global and local vectors.
    local_range = range(global_size)
    is_seq = PETSc.IS().createGeneral(local_range, comm=COMM_SELF)
    is_global = PETSc.IS().createGeneral(local_range, comm=COMM_WORLD)

    # Create scatter context and perform scatter operation.
    scatter_ctx = PETSc.Scatter().create(v_global, is_global, v_seq, is_seq)
    is_global.destroy()
    is_seq.destroy()
    # scatter_ctx.scatter(v_global, v_seq, mode=PETSc.ScatterMode.FORWARD)
    scatter_ctx.scatter(v_global, v_seq, mode=0)

    scatter_ctx.destroy()

    return v_seq


matvec_start = time.time()

# --------------------------------------------
# EXP: Multiplication of the transpose of mpi PETSc matrix with a sequential PETSc vector
# c = A.T * b
# [k x 1] = [m x k].T * [m x 1]
# --------------------------------------------

m, k = 10000, 50
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.randint(low=0, high=6, size=(m, k))
b_np = np.random.randint(low=0, high=6, size=m)

# Create B as a sequential matrix on each process
A = create_petsc_matrix(A_np)
print_matrix_partitioning(A, "matrix A")
b = create_petsc_vector(b_np)
print_vector_partitioning(b, "vector b")

A_local = get_local_submatrix(A)
print_matrix_partitioning(A_local, "matrix A_local")
b_local = get_local_subvector(b)
print_vector_partitioning(b_local, "vector b_local")

matvec_setup = time.time()
matvec_time = matvec_setup - matvec_start
matvec_time_avg = COMM_WORLD.allreduce(matvec_time, op=MPI.SUM) / nproc
Print(
    f"-Setup A matrix and b vector: {matvec_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

c_local = create_petsc_vector_seq(np.zeros(k))
print_vector_partitioning(c_local, "c_local")

A_local.multTranspose(b_local, c_local)

c_seq_array = COMM_WORLD.allreduce(c_local.getArray(), op=MPI.SUM)
c_seq = create_petsc_vector_seq(c_seq_array)
print_vector_partitioning(c_seq, "c_seq")

matvec_arom = time.time()
matvec_time = matvec_arom - matvec_setup
matvec_time_avg = COMM_WORLD.allreduce(matvec_time, op=MPI.SUM) / nproc
Print(
    f"-Compute C = A.T * b: {matvec_time_avg: 2.2f} s",
    Fore.MAGENTA,
)

# --------------------------------------------
# TEST: Multiplication of a numpy matrix and a numpy vector
# --------------------------------------------
Ab_np = np.dot(A_np.T, b_np.flatten())
Print(f"Vector Ab_np [{Ab_np.shape[0]}]")
Print(f"{Ab_np}")

# Get the local values from C
# local_rows_start, local_rows_end = c.getOwnershipRange()
c_test = c_seq.getArray()[:]

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(c_test, Ab_np[:], decimal=5)
