"""Experimenting with petsc mat-mat multiplication"""
# import pdb

import numpy as np
from firedrake import COMM_WORLD
from firedrake.petsc import PETSc
from numpy.testing import assert_array_almost_equal

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def Print(x: str):
    """Prints the string only on the root process

    Args:
        x (str): String to be printed
    """
    PETSc.Sys.Print(x)


def create_petsc_matrix_non_partitioned(input_array, sparse=True):
    """Building a mpi non-partitioned petsc matrix from an array

    Args:
        input_array (np array): Input array
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        mpi mat: PETSc matrix
    """
    # Create a sparse or dense matrix based on the 'sparse' argument
    assert len(input_array.shape) == 2

    m, n = input_array.shape

    Print(f"The non-partitioned petsc matrix to be created is of size: {m, n}")

    # If the size of the input vector is 1, handle it as a special case
    if m == 1 and n == 1:
        if sparse:
            matrix = PETSc.Mat().createAIJ(size=((1, 1), (1, 1)), comm=COMM_WORLD)
        else:
            matrix = PETSc.Mat().createDense(size=((1, 1), (1, 1)), comm=COMM_WORLD)
        matrix.setUp()

        if rank == 0:
            matrix.setValue(0, 0, input_array[0, 0])

        matrix.assemblyBegin()
        matrix.assemblyEnd()
        return matrix

    if sparse:
        matrix = PETSc.Mat().createAIJ(size=((m, n), (m, n)), comm=COMM_WORLD)
    else:
        matrix = PETSc.Mat().createDense(size=((m, n), (m, n)), comm=COMM_WORLD)
    matrix.setUp()

    # Set the values of the matrix
    matrix.setValues(range(m), range(n), input_array[:, :], addv=False)

    # Assembly the matrix to compute the final structure
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


def create_petsc_matrix(input_array, partition_like=None, sparse=True):
    """Create a PETSc matrix from an input_array

    Args:
        input_array (np array): Input array
        partition_like (petsc mat, optional): Petsc matrix. Defaults to None.
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        petsc mat: PETSc matrix
    """
    # Check if input_array is 1D and reshape if necessary
    assert len(input_array.shape) == 2, "Input array should be 2-dimensional"
    global_rows, global_cols = input_array.shape

    Print("Creating mpi matrix")
    comm = COMM_WORLD
    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        # No parallelization in the columns, set local_cols = None to parallelize
        size = ((local_rows, global_rows), (global_cols, global_cols))
    else:
        size = ((None, global_rows), (global_cols, global_cols))

    # Create a sparse or dense matrix based on the 'sparse' argument
    if sparse:
        matrix = PETSc.Mat().createAIJ(size=size, comm=comm)
    else:
        matrix = PETSc.Mat().createDense(size=size, comm=comm)
    matrix.setUp()

    local_rows_start, local_rows_end = matrix.getOwnershipRange()
    local_rows = local_rows_end - local_rows_start

    print(
        f"For proc {rank} local_rows = {local_rows}: ({local_rows_start}, {local_rows_end})"
    )

    for counter, i in enumerate(range(local_rows_start, local_rows_end)):
        # Calculate the correct row in the array for the current process
        row_in_array = counter + local_rows_start
        matrix.setValues(
            i, range(global_cols), input_array[row_in_array, :], addv=False
        )

    # Assembly the matrix to compute the final structure
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


m, k = 10, 5
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
# A_np = np.random.rand(m, k)
# B_np = np.random.rand(k, k)
A_np = np.random.randint(low=0, high=6, size=(m, k))
B_np = np.random.randint(low=0, high=6, size=(k, k))


A = create_petsc_matrix(A_np)
Print("MATRIX A")
Print(A.getType())
Print(A.getSizes())
A.view()
Print("")

# pdb.set_trace()

B = create_petsc_matrix_non_partitioned(B_np)
Print("MATRIX B")
Print(B.getType())
Print(B.getSizes())
B.view()
Print("")


# pdb.set_trace()

# C = A.matMult(B)
C = A * B

Print("MATRIX C")
Print(C.getType())
Print(C.getSizes())
C.view()

# Compute the product using numpy and check the result only on root process
AB_np = np.dot(A_np, B_np)
Print(AB_np)

# # Get the local ranges for C
local_rows_start, local_rows_end = C.getOwnershipRange()

# # Get the local values from C
C_local = C.getValues(range(local_rows_start, local_rows_end), range(k))

print(f"For rank {rank} C_local: {C_local}")
print(
    f"For rank {rank} AB_np[local_rows_start:local_rows_end,:]: {AB_np[local_rows_start:local_rows_end,:]}"
)

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(C_local, AB_np[local_rows_start:local_rows_end, :], decimal=5)
