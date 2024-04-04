"""Experimenting with PETSc mat-vec multiplication"""

import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD
from mpi4py import MPI
from numpy.testing import assert_array_almost_equal

from utilities import (
    Print,
    create_petsc_matrix,
    create_petsc_vector,
    create_petsc_vector_seq,
    get_local_submatrix,
    get_local_subvector,
)

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


matvec_start = time.time()

# --------------------------------------------
# EXP: Multiplication of the transpose of mpi PETSc matrix with a sequential PETSc vector
# c = A.T * b
# [k x 1] = [m x k].T * [m x 1]
# --------------------------------------------


def matvec_transpose_multiplication(A, b):
    """
    Multiply the transpose of a distributed matrix A with a distributed vector b.
    The computation is performed locally on each process and the results are aggregated.

    Args:
        A (PETSc Mat): Distributed MPI PETSc matrix [mxk].
        b (PETSc Vec): Distributed MPI PETSc vector [mx1].


    Returns:
        PETSc Vec: The result of A.T * b as a sequential PETSc vector [kx1].
    """
    start_time = time.time()
    A_local = get_local_submatrix(A)
    b_local = get_local_subvector(b)

    k = A.getSize()[1]

    c_local = create_petsc_vector_seq(np.zeros(k))

    A_local.multTranspose(b_local, c_local)

    # Aggregate the results from all processes
    c_seq_array = COMM_WORLD.allreduce(c_local.getArray(), op=MPI.SUM)
    c_seq = create_petsc_vector_seq(c_seq_array)

    end_time = time.time()
    duration_local = end_time - start_time
    duration = COMM_WORLD.allreduce(duration_local, op=MPI.SUM)
    Print(
        f"Time taken for matvec_transpose_multiplication: {duration}",
        Fore.GREEN,
    )

    return c_seq


def matvec_transpose_multiplication_v2(A, b):
    """
    Multiply the transpose of a distributed matrix A with a distributed vector b by
    performing k vector-vector multiplications. Store the result in a sequential vector c.

    Args:
        A (PETSc Mat): Distributed MPI PETSc matrix [mxk].
        b (PETSc Vec): Distributed MPI PETSc vector [mx1].

    Returns:
        PETSc Vec: The result of A.T * b as a sequential PETSc vector [kx1].
    """
    start_time = time.time()
    k = A.getSize()[1]

    c_seq = create_petsc_vector_seq(np.zeros(k))

    for i in range(k):

        loop_start = time.time()

        # Extract the i-th column of A as a vector
        a_i = A.getColumnVector(i)

        # Perform vector-vector multiplication: a_i.T * b
        c_i = a_i.dot(b)

        # Set the i-th value of the resulting vector c
        c_seq.setValue(i, c_i, addv=False)  # addv=False to insert the value directly

        loop_end = time.time()
        loop_duration_local = loop_end - loop_start
        loop_duration = COMM_WORLD.allreduce(loop_duration_local, op=MPI.SUM)
        Print(f"Time taken for loop {i}: {loop_duration}", Fore.YELLOW)

    # Finalize the assembly of c_seq to ensure all values are correctly set
    c_seq.assemblyBegin()
    c_seq.assemblyEnd()

    end_time = time.time()
    duration_local = end_time - start_time
    duration = COMM_WORLD.allreduce(duration_local, op=MPI.SUM)
    Print(
        f"Time taken for matvec_transpose_multiplication_v2: {duration}",
        Fore.RED,
    )

    return c_seq


def test_matvec_transpose_multiplication(A_np, b_np, c_seq):
    """
    Test the multiplication of the transpose of a numpy matrix A_np with a numpy vector b_np
    against the PETSc-based result c_seq.

    Args:
        A_np (np.array): The numpy representation of the matrix.
        b_np (np.array): The numpy representation of the vector.
        c_seq (PETSc Vec): The result of A.T * b using PETSc.
    """
    # Multiplication using numpy for comparison
    Ab_np = np.dot(A_np.T, b_np.flatten())
    Print(f"Vector Ab_np [{Ab_np.shape[0]}]")
    # Print(f"{Ab_np}")

    # Get the array from the PETSc vector
    c_test = c_seq.getArray()[:]

    # Assert the correctness of the multiplication
    assert_array_almost_equal(c_test, Ab_np[:], decimal=5)


if __name__ == "__main__":
    m, k = 5000000, 100
    np.random.seed(0)
    A_np = np.random.randint(low=0, high=6, size=(m, k))
    b_np = np.random.randint(low=0, high=6, size=m)

    # Setup
    A = create_petsc_matrix(A_np)
    # print_matrix_partitioning(A, "matrix A")
    b = create_petsc_vector(b_np)
    # print_vector_partitioning(b, "vector b")

    # Perform the operation using the first implementation
    Print("Testing original matvec_transpose_multiplication", Fore.BLUE)
    c_seq = matvec_transpose_multiplication(A, b)
    # print_vector_partitioning(c_seq, "c_seq from original implementation")

    # Test the result of the first implementation
    test_matvec_transpose_multiplication(A_np, b_np, c_seq)

    # Perform the operation using the new implementation
    Print("Testing new matvec_transpose_multiplication_v2", Fore.BLUE)
    c_seq_v2 = matvec_transpose_multiplication_v2(A, b)
    # print_vector_partitioning(c_seq_v2, "c_seq from new implementation")

    # Test the result of the new implementation
    test_matvec_transpose_multiplication(A_np, b_np, c_seq_v2)
