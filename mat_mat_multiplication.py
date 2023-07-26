"""Experimenting with petsc mat-mat multiplication"""
# import pdb

import numpy as np
from firedrake import COMM_WORLD
from firedrake.petsc import PETSc

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def create_matrix(input_array, sparse=True):
    """Building a seq petsc matrix from an array.

    Args:
        input_array (np array): Input array
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        seq mat: PETSc matrix
    """
    assert len(input_array.shape) == 2, "Input array should be 2-dimensional"

    m, n = input_array.shape

    # Create a sparse or dense matrix based on the 'sparse' argument
    if sparse:
        matrix = PETSc.Mat().createAIJ(size=(m, n), comm=PETSc.COMM_SELF)
    else:
        matrix = PETSc.Mat().createDense(size=(m, n), comm=PETSc.COMM_SELF)

    matrix.setType("aij")

    matrix.setUp()
    matrix.setValues(range(m), range(n), input_array)
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


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

    print(f"The non-partitioned petsc matrix to be created is of size: {m, n}")

    # If the size of the input vector is 1, handle it as a special case
    if m == 1 and n == 1:
        if sparse:
            matrix = PETSc.Mat().createAIJ(size=((1, 1), (1, 1)), comm=COMM_WORLD)
        else:
            matrix = PETSc.Mat().createDense(size=((1, 1), (1, 1)), comm=COMM_WORLD)
        matrix.setUp()

        local_start, local_end = matrix.getOwnershipRange()
        # print(f"For proc {rank} matrix OwnershipRange: {local_start, local_end }")
        # print(f"for proc {rank} input_array: {input_array}")
        # pdb.set_trace()

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

    local_start, local_end = matrix.getOwnershipRange()
    # print(f"For proc {rank} matrix OwnershipRange: {local_start, local_end}")
    # print(f"for proc {rank} input_array: {input_array}")
    # pdb.set_trace()

    # Explicitly define dtype as np.int32
    matrix.setValues(range(m), range(n), input_array[:, :], addv=False)

    # pdb.set_trace()
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

    # If the size of the matrix is smaller than the number of processes, use a single process
    if global_rows < nproc:
        print("Creating sequential matrix")
        comm = PETSc.COMM_SELF
        size = (global_rows, global_cols)
    else:
        print("Creating mpi matrix")
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


def print_local_size(matrix, rank):
    """Print the local size of the matrix

    Args:
        matrix (petsc mat): PETSc matrix
        rank (int): process rank
    """
    local_rows, _ = matrix.getSizes()
    print(f"Process {rank} local size: {local_rows}")


A = create_petsc_matrix(np.array([[1, 2], [3, 4]]))
PETSc.Sys.Print("MATRIX A")
print(A.getType())
print(A.getSizes())
A.view()
PETSc.Sys.Print("")

# pdb.set_trace()

B = create_petsc_matrix_non_partitioned(np.array([[1, 0], [0, 1]]))
PETSc.Sys.Print("MATRIX B")
print(B.getType())
print(B.getSizes())
B.view()
PETSc.Sys.Print("")


# pdb.set_trace()

# C = A.matMult(B)
C = A * B

PETSc.Sys.Print("MATRIX C")
print(C.getType())
print(C.getSizes())
C.view()
