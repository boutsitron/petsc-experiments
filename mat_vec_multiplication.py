"""Experimenting with PETSc mat-vec multiplication"""

import numpy as np
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc
from numpy.testing import assert_array_almost_equal

from utilities import (
    Print,
    convert_global_matrix_to_seq,
    create_petsc_matrix,
    create_petsc_vector,
    create_petsc_vector_seq,
    print_matrix_partitioning,
    print_vector_partitioning,
)

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def convert_global_vector_to_seq(v_global):
    """
    Fetch the complete global vector `v_global` to a local sequential vector on all ranks.

    Args:
        v_global (PETSc mpi Vec): Global MPI PETSc vector.

    Returns:
        PETSc seq Vec: Local sequential vector having the complete values of `v_global`.
    """
    # Create a local sequential vector of the same size as the global vector.
    global_size = v_global.getSize()
    v_seq = PETSc.Vec().createSeq(global_size, comm=COMM_SELF)

    # Create index sets for global and local vectors.
    local_range = range(global_size)
    is_local_complete = PETSc.IS().createGeneral(local_range, comm=COMM_SELF)
    is_global = PETSc.IS().createGeneral(local_range, comm=COMM_WORLD)

    # Step 4: Create scatter context and perform scatter operation.
    scatter_ctx = PETSc.Scatter().create(v_global, is_global, v_seq, is_local_complete)
    # scatter_ctx.scatter(v_global, v_seq, mode=PETSc.ScatterMode.FORWARD)
    scatter_ctx.scatter(v_global, v_seq, mode=0)

    return v_seq


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


# --------------------------------------------
# EXP: Multiplication of the transpose of mpi PETSc matrix with a sequential PETSc vector
# c = A.T * b
# [k x 1] = [m x k].T * [m x 1]
# --------------------------------------------

m, k = 11, 3
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.randint(low=0, high=6, size=(m, k))
b_np = np.random.randint(low=0, high=6, size=m)

# Create B as a sequential matrix on each process
A = create_petsc_matrix(A_np)
print_matrix_partitioning(A, "matrix A")
b = create_petsc_vector(b_np)
print_vector_partitioning(b, "vector b")

A_seq = convert_global_matrix_to_seq(A)
print_matrix_partitioning(A_seq, "matrix A_seq")
b_seq = convert_global_vector_to_seq(b)
print_vector_partitioning(b_seq, "vector b_seq")

c_seq = create_petsc_vector_seq(np.zeros(k))
A_seq.multTranspose(b_seq, c_seq)

print_vector_partitioning(c_seq, "c_seq")

# c = scatter_local_to_global_vector(c_seq)
# print_vector_partitioning(c, "c")

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
