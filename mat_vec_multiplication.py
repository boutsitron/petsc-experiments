"""Experimenting with PETSc mat-mat multiplication"""

import numpy as np
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc
from numpy.testing import assert_array_almost_equal

from utilities import Print, print_vector_partitioning

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def create_petsc_vector(input_array):
    """Create a PETSc sequential vector from an input array

    Args:
        input_array (np array): Input 1-dimensional array

    Returns:
        PETSc Vec: PETSc sequential vector
    """
    # Check if input_array is 1D and reshape if necessary
    if len(input_array.shape) != 1:
        raise ValueError("Input array should be 1-dimensional")

    m = input_array.shape[0]

    # Create a sequential vector
    vector = PETSc.Vec().createMPI(size=m, comm=COMM_WORLD)

    # Set the values
    vector.setValues(range(m), input_array)

    # Assembly the vector to compute the final structure
    vector.assemblyBegin()
    vector.assemblyEnd()

    return vector


def create_petsc_matrix(input_array, partition_like=None, sparse=True):
    """Create a PETSc matrix from an input_array

    Args:
        input_array (np array): Input array
        partition_like (PETSc mat, optional): Petsc matrix. Defaults to None.
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        PETSc mat: PETSc matrix
    """
    # Check if input_array is 1D and reshape if necessary
    assert len(input_array.shape) == 2, "Input array should be 2-dimensional"
    global_rows, global_cols = input_array.shape

    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        # No parallelization in the columns, set local_cols = None to parallelize
        size = ((local_rows, global_rows), (global_cols, global_cols))
    else:
        size = ((None, global_rows), (global_cols, global_cols))

    # Create a sparse or dense matrix based on the 'sparse' argument
    if sparse:
        matrix = PETSc.Mat().createAIJ(size=size, comm=COMM_WORLD)
    else:
        matrix = PETSc.Mat().createDense(size=size, comm=COMM_WORLD)
    matrix.setUp()

    local_rows_start, local_rows_end = matrix.getOwnershipRange()

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


def multiply_matrix_transpose_to_vector(A, b):
    """Multiply 2 sequential matrices

    Args:
        A (mpiaij): local submatrix of A
        b (mpivec): sequential vector

    Returns:
        seq vec: PETSc vector that is the result of the multiplication of A.T and b
    """
    (A_local_rows, A_rows), (_, A_cols) = A.getSizes()
    b_size = b.getSize()
    # b.view()
    assert (
        A_rows == b_size
    ), f"Incompatible global matrix/vector sizes for multiplication: {A_rows} != {b_size}"

    # Assert that local dimensions match for multiplication
    assert (
        A_local_rows == b.getSizes()[0]
    ), f"Incompatible local sizes: {A_local_rows} != {b.getSizes()[0]}"

    # c_local [kx1] <- A.T [kxm] * b [mx1]
    c_local = PETSc.Vec().createSeq(size=A_cols, comm=COMM_SELF)

    A.multTranspose(b, c_local)
    return c_local


def scatter_local_to_global_vector(v_local):
    """Create a global MPI PETSc vector from the local sequential vector

    Args:
        v_local (PETSc Vec): local sequential vector

    Returns:
        mpi PETSc Vec: global MPI vector
    """
    # Get the global size of the vector, which is the sum of the sizes of all the local vectors across all MPI ranks.
    global_size = v_local.getSize()
    # Get the total size of the local vector. This is the size of the vector on the current MPI rank.
    local_size = len(v_local.getArray())

    v_global = PETSc.Vec().createMPI(size=(local_size, global_size), comm=COMM_WORLD)

    # Create an index set for the local/global vector.
    is_local = PETSc.IS().createGeneral(range(local_size), comm=COMM_SELF)
    is_global = PETSc.IS().createGeneral(range(local_size), comm=COMM_WORLD)

    # Create a scatter context that will allow us to scatter the values from the local vector to the global vector.
    scatter_ctx = PETSc.Scatter().create(v_local, is_local, v_global, is_global)

    # Perform the scattering operation.
    # mode=1 refers to the forward scatter. It takes the values in the local vector and puts them in the appropriate places in the global vector.
    scatter_ctx.scatter(v_local, v_global, mode=1)

    # Assembly the global vector to finalize its structure
    v_global.assemblyBegin()
    v_global.assemblyEnd()

    return v_global


def get_complete_vec_local(v_global):
    """
    Fetch the complete global vector `v_global` to a local sequential vector on all ranks.

    Args:
        v_global (PETSc Vec): Global MPI PETSc vector.

    Returns:
        PETSc Vec: Local sequential vector having the complete values of `v_global`.
    """
    # Create a local sequential vector of the same size as the global vector.
    global_size = v_global.getSize()
    v_local_complete = PETSc.Vec().createSeq(global_size, comm=COMM_SELF)

    # Create index sets for global and local vectors.
    local_range = range(global_size)
    is_local_complete = PETSc.IS().createGeneral(local_range, comm=COMM_SELF)
    is_global = PETSc.IS().createGeneral(local_range, comm=COMM_WORLD)

    # Step 4: Create scatter context and perform scatter operation.
    scatter_ctx = PETSc.Scatter().create(
        v_global, is_global, v_local_complete, is_local_complete
    )
    # scatter_ctx.scatter(v_global, v_local_complete, mode=PETSc.ScatterMode.FORWARD)
    scatter_ctx.scatter(v_global, v_local_complete, mode=0)

    return v_local_complete


# --------------------------------------------
# TEST: Multiplication of an mpi PETSc matrix with a sequential PETSc vector
#  c = A.T * b
# [k x 1] = [m x k].T * [m x 1]
# --------------------------------------------

m, k = 11, 7
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.randint(low=0, high=6, size=(m, k))
b_np = np.random.randint(low=0, high=6, size=m)

# Create B as a sequential matrix on each process
b = create_petsc_vector(b_np)
# print_vector_partitioning(b, "vector b")

A = create_petsc_matrix(A_np)
# print_matrix_partitioning(A, "matrix A")
# print_matrix_partitioning(A, "matrix A")

# Multiplication of 2 sequential matrices
c_local = multiply_matrix_transpose_to_vector(A, b)
# print_vector_partitioning(c_local, "vector c_local")

c = scatter_local_to_global_vector(c_local)
# print_vector_partitioning(c, "vector c")

# Assuming c is your global PETSc vector
c_local_complete = get_complete_vec_local(c)
print_vector_partitioning(c_local_complete, "vector c_local_complete")

# --------------------------------------------
# TEST: Multiplication of a numpy matrix and a numpy vector
# --------------------------------------------
Ab_np = np.dot(A_np.T, b_np.flatten())
Print(f"Vector Ab_np [{Ab_np.shape[0]}]")
Print(Ab_np)

# Get the local values from C
local_rows_start, local_rows_end = c.getOwnershipRange()
c_test = c.getArray()[local_rows_start:local_rows_end]
print(f"For rank {rank}: c_test {c_test}")

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(c_test, Ab_np[local_rows_start:local_rows_end], decimal=5)
