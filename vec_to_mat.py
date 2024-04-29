"""Experimenting with PETSc mat-vec multiplication"""

import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI

from utilities import (
    Print,
    convert_petsc_vector_to_matrix_seq,
    create_petsc_vector,
    gather_local_to_global_matrix,
    get_local_subvector,
)

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


# --------------------------------------------
# EXP: Multiplication of the transpose of mpi PETSc matrix with a sequential PETSc vector
# c = A.T * b
# [k x 1] = [m x k].T * [m x 1]
# --------------------------------------------


def convert_petsc_vector_to_matrix(vector: PETSc.Vec) -> PETSc.Mat:
    """Convert a PETSc vector to a PETSc matrix of one column
    with the same number of rows as the size of the vector

    Args:
        vector (PETSc.Vec): PETSc vector

    Returns:
        PETSc.Mat: PETSc matrix (partitioned)
    """
    vector_local = get_local_subvector(vector)
    matrix_local = convert_petsc_vector_to_matrix_seq(vector_local)
    return gather_local_to_global_matrix(matrix_local, partition_like=vector)


def test_matrix_vector_equivalence(vector: PETSc.Vec, matrix: PETSc.Mat):
    """Test if the PETSc matrix and PETSc vector contain the same values within the ownership range of each process."""
    local_range = vector.getOwnershipRange()
    local_values_vector = vector.getArray()

    # Get the corresponding column values from the matrix
    local_values_matrix = np.array(
        [matrix.getValue(row, 0) for row in range(*local_range)]
    )

    # Check if both arrays are equivalent
    assert np.allclose(
        local_values_vector, local_values_matrix
    ), f"Values in vector and matrix do not match on process {rank}."


if __name__ == "__main__":
    m, k = 500000, 50
    np.random.seed(0)
    b_np = np.random.randint(low=0, high=6, size=m)

    # Setup
    b = create_petsc_vector(b_np)
    # print_vector_partitioning(b, "vector b")

    matvec_start = time.time()

    # Convert the vector to a matrix
    b_matrix = convert_petsc_vector_to_matrix(b)

    # Run the test
    test_matrix_vector_equivalence(b, b_matrix)

    print(f"Test completed successfully on process {rank}.")
    matvec_end = time.time()

    matvec_local = matvec_end - matvec_start
    matvec = COMM_WORLD.allreduce(matvec_local, op=MPI.SUM)
    Print(
        f"Total time taken for mat-vec operations:: {matvec:.2f} seconds",
        Fore.RED,
    )
