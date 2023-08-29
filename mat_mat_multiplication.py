"""Experimenting with PETSc mat-mat multiplication"""

import numpy as np
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI
from numpy.testing import assert_array_almost_equal

from utilities import (
    Print,
    create_petsc_matrix,
    get_local_submatrix,
    print_matrix_partitioning,
)

# import pdb

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def create_petsc_matrix_seq(input_array):
    """Building a sequential PETSc matrix from an array

    Args:
        input_array (np array): Input array

    Returns:
        seq mat: PETSc matrix
    """
    assert len(input_array.shape) == 2

    m, n = input_array.shape
    matrix = PETSc.Mat().createAIJ(size=(m, n), comm=COMM_SELF)
    matrix.setUp()

    matrix.setValues(range(m), range(n), input_array, addv=False)

    # Assembly the matrix to compute the final structure
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


def multiply_matrices_seq(A_seq, B_seq):
    """Multiply 2 sequential matrices

    Args:
        A_seq (seqaij): local submatrix of A
        B_seq (seqaij): sequential matrix B

    Returns:
        seq mat: PETSc matrix that is the product of A_seq and B_seq
    """
    _, A_seq_cols = A_seq.getSize()
    B_seq_rows, _ = B_seq.getSize()
    assert (
        A_seq_cols == B_seq_rows
    ), f"Incompatible matrix sizes for multiplication: {A_seq_cols} != {B_seq_rows}"
    C_local = A_seq.matMult(B_seq)
    return C_local


def concatenate_local_to_global_matrix(C_local):
    """Create the global matrix C from the local submatrix C_local

    Args:
        C_local (seqaij): local submatrix of C

    Returns:
        mpi PETSc mat: partitioned PETSc matrix C
    """
    C_local_rows, C_local_cols = C_local.getSize()

    # Get the global number of rows for C matrix using MPI Allreduce
    global_rows = COMM_WORLD.allreduce(C_local_rows, op=MPI.SUM)

    C = PETSc.Mat().createAIJ(
        size=((None, global_rows), (C_local_cols, C_local_cols)), comm=COMM_WORLD
    )
    C.setUp()

    # The exscan operation is used to get the starting global row for each process.
    # The result of the exclusive scan is the sum of the local rows from previous ranks.
    global_row_start = COMM_WORLD.exscan(C_local_rows, op=MPI.SUM)
    if rank == 0:
        global_row_start = 0

    for i in range(C_local_rows):
        cols, values = C_local.getRow(i)
        global_row = i + global_row_start
        C.setValues(global_row, cols, values)

    C.assemblyBegin()
    C.assemblyEnd()

    return C


# --------------------------------------------
# EXP: Multiplication of an mpi PETSc matrix with a sequential PETSc matrix
#  C = A * B
# [m x k] = [m x k] * [k x k]
# --------------------------------------------

m, k = 11, 7
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.randint(low=0, high=6, size=(m, k))
B_np = np.random.randint(low=0, high=6, size=(k, k))

# Create B as a sequential matrix on each process
B_seq = create_petsc_matrix_seq(B_np)
print_matrix_partitioning(B_seq, "B")

A = create_petsc_matrix(A_np)
print_matrix_partitioning(A, "A")

# Getting the correct local submatrix to be multiplied by B_seq
A_local = get_local_submatrix(A)

# Multiplication of 2 sequential matrices
C_local = multiply_matrices_seq(A_local, B_seq)

# Creating the global C matrix
C = concatenate_local_to_global_matrix(C_local)
print_matrix_partitioning(C, "C")

# --------------------------------------------
# TEST: Multiplication of 2 numpy matrices
# --------------------------------------------
AB_np = np.dot(A_np, B_np)
Print(f"MATRIX AB_np [{AB_np.shape[0]}x{AB_np.shape[1]}]")
Print(AB_np)

# Get the local values from C
local_rows_start, local_rows_end = C.getOwnershipRange()
C_local = C.getValues(range(local_rows_start, local_rows_end), range(k))

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(C_local, AB_np[local_rows_start:local_rows_end, :], decimal=5)
