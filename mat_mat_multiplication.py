"""Experimenting with PETSc mat-mat multiplication"""

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
    get_local_submatrix,
    print_matrix_partitioning,
)

size = COMM_WORLD.size
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


def concatenate_local_to_global_matrix(
    local_matrix, partition_like=None, mat_type=None
):
    """Create the global matrix C from the local submatrix local_matrix

    Args:
        local_matrix (seqaij): local submatrix of global_matrix
        partition_like (mpiaij): partitioned PETSc matrix
        mat_type (str): type of the global matrix. Defaults to None. If None, the type of local_matrix is used.

    Returns:
        mpi PETSc mat: partitioned PETSc matrix
    """
    local_matrix_rows, local_matrix_cols = local_matrix.getSize()
    global_rows = COMM_WORLD.allreduce(local_matrix_rows, op=MPI.SUM)

    print(f"Local matrix size {local_matrix_rows}x{local_matrix_cols}")
    print(f"Global matrix size: {global_rows}x{local_matrix_cols}")

    # Determine the local portion of the vector
    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        size = ((local_rows, global_rows), (local_matrix_cols, local_matrix_cols))
    else:
        size = ((None, global_rows), (local_matrix_cols, local_matrix_cols))

    if mat_type is None:
        mat_type = local_matrix.getType()

    if "dense" in mat_type:
        sparse = False
    else:
        sparse = True

    if sparse:
        global_matrix = PETSc.Mat().createAIJ(size=size, comm=COMM_WORLD)
    else:
        global_matrix = PETSc.Mat().createDense(size=size, comm=COMM_WORLD)
    global_matrix.setUp()

    # The exscan operation is used to get the starting global row for each process.
    # The result of the exclusive scan is the sum of the local rows from previous ranks.
    global_row_start = COMM_WORLD.exscan(local_matrix_rows, op=MPI.SUM)
    if rank == 0:
        global_row_start = 0

    concatenate_start_3 = time.time()

    # for i in range(local_matrix_rows):
    #     cols, values = local_matrix.getRow(i)
    #     global_row = i + global_row_start
    #     global_matrix.setValues(global_row, cols, values)

    all_cols = []
    all_values = []
    all_global_rows = [i + global_row_start for i in range(local_matrix_rows)]

    for i in range(len(all_global_rows)):
        cols, values = local_matrix.getRow(i)
        # print(f"cols: {cols}, values: {values}")
        all_cols.append(cols)
        all_values.append(values)

    print(len(all_cols), all_cols[0])
    print(len(all_values), all_values[0])
    print(len(all_global_rows), all_global_rows[0])
    print()

    for j in range(local_matrix_cols):
        columns = [all_cols[i][j] for i in range(len(all_cols))]
        values = [all_values[i][j] for i in range(len(all_values))]

        print(np.shape(columns), columns[0])
        print(np.shape(values), values[0])
        print(np.shape(all_global_rows), all_global_rows[0])

        global_matrix.setValues(all_global_rows, columns, values)

    concatenate_start_4 = time.time()
    Print(
        f"  -Setting values: {concatenate_start_4 - concatenate_start_3: 2.2f} s",
        Fore.GREEN,
    )

    global_matrix.assemblyBegin()
    global_matrix.assemblyEnd()

    return global_matrix


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
C = concatenate_local_to_global_matrix(C_local) if size > 1 else C_local
print_matrix_partitioning(C, "C")

# --------------------------------------------
# TEST: Multiplication of 2 numpy matrices
# --------------------------------------------
AB_np = np.dot(A_np, B_np)
Print(f"MATRIX AB_np [{AB_np.shape[0]}x{AB_np.shape[1]}]")
Print(f"{AB_np}")

# Get the local values from C
local_rows_start, local_rows_end = C.getOwnershipRange()
C_local = C.getValues(range(local_rows_start, local_rows_end), range(k))

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(C_local, AB_np[local_rows_start:local_rows_end, :], decimal=5)
