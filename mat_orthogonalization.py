"""Experimenting with SLEPc orthogonalization"""

import contextlib
import sys
import time

import numpy as np
from colorama import Fore
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc

from utilities import (
    Print,
    create_petsc_matrix,
    get_local_submatrix,
    print_matrix_partitioning,
)

with contextlib.suppress(ImportError):
    import slepc4py

    slepc4py.init(sys.argv)
    from slepc4py import SLEPc

from numpy.testing import assert_array_almost_equal

rank = COMM_WORLD.rank
EPSILON_SVD = 1e-4
EPS = sys.float_info.epsilon


def convert_global_matrix_to_seq(A):
    """Convert a partitioned matrix to a sequential one such that each processor holds a duplicate of the full matrix.

    Args:
        A (PETSc.Mat): The partitioned matrix

    Returns:
        PETSc.Mat: The sequential matrix
    """
    # Step 1: Get the local submatrix
    A_local = get_local_submatrix(A)

    # Step 2: Convert the local submatrix to numpy array
    A_local_rows, A_local_cols = A_local.getSize()
    A_local_array = np.zeros((A_local_rows, A_local_cols))
    for i in range(A_local_rows):
        _, values = A_local.getRow(i)
        A_local_array[i, :] = values

    # Step 3: Use allgather to collect all local matrices to all processes
    gathered_data = COMM_WORLD.allgather(A_local_array)

    # Step 4: Stack the local matrices to create the full sequential matrix
    full_matrix_array = np.vstack(gathered_data)

    # Step 5: Create a new sequential PETSc matrix
    m, n = full_matrix_array.shape
    A_seq = PETSc.Mat().createDense([m, n], comm=COMM_SELF)
    A_seq.setUp()

    for i in range(m):
        A_seq.setValues(i, list(range(n)), full_matrix_array[i, :])

    A_seq.assemblyBegin()
    A_seq.assemblyEnd()

    return A_seq


def orthogonality(PPhi):  # sourcery skip: avoid-builtin-shadow
    """Checking and correcting orthogonality

    Args:
        PPhi (PETSc.Mat): Matrix of size [m x k+1].

    Returns:
        PETSc.Mat: Matrix of size [m x k+1].
    """
    # checking orthogonality
    orth_start = time.time()

    m, kp1 = PPhi.getSize()

    Phi1 = PPhi.getColumnVector(0)
    Phi2 = PPhi.getColumnVector(kp1 - 1)

    # Compute dot product using PETSc function
    dot_product = Phi1.dot(Phi2)

    if abs(dot_product) > min(EPSILON_SVD, EPS * m):
        Print("    Basis has lost (numerical) orthogonality", Fore.RED)

        # Type can be CHOL, GS, mro(), SVQB, TSQR, TSQRCHOL
        _type = SLEPc.BV().OrthogBlockType.GS

        # Check if the matrix is dense
        mat_type = PPhi.getType()

        if "dense" in mat_type:
            bv = SLEPc.BV().createFromMat(PPhi)
        elif "seq" not in mat_type:
            PPhi_seq = convert_global_matrix_to_seq(PPhi)
            bv = SLEPc.BV().createFromMat(PPhi_seq.convert("dense"))
        else:
            bv = SLEPc.BV().createFromMat(PPhi.convert("dense"))
        bv.setFromOptions()
        bv.setOrthogonalization(_type)
        bv.orthogonalize()

        PPhi = bv.createMat()

        # # Assembly the matrix to compute the final structure
        if not PPhi.assembled:
            PPhi.assemblyBegin()
            PPhi.assemblyEnd()
    else:
        Print("    Basis is orthogonal", Fore.GREEN)

    # Checking and correcting orthogonality
    Print(f"    -Orthogonality: {time.time() - orth_start: 2.2f} s", Fore.GREEN)

    return PPhi


# --------------------------------------------
# EXP: Orthogonalization of an mpi PETSc matrix
# --------------------------------------------

m, k = 11, 7
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.randint(low=0, high=6, size=(m, k))

A = create_petsc_matrix(A_np, sparse=False)
print_matrix_partitioning(A, "A")

A_orthogonal = orthogonality(A)
print_matrix_partitioning(A_orthogonal, "A_orthogonal", values=False)

# --------------------------------------------
# TEST: Orthogonalization of a numpy matrix
# --------------------------------------------
# Generate A_np_orthogonal
A_np_orthogonal, _ = np.linalg.qr(A_np)

# Get the local values from A_orthogonal
local_rows_start, local_rows_end = A_orthogonal.getOwnershipRange()
A_orthogonal_local = A_orthogonal.getValues(
    range(local_rows_start, local_rows_end), range(k)
)

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(
    np.abs(A_orthogonal_local),
    np.abs(A_np_orthogonal[local_rows_start:local_rows_end, :]),
    decimal=5,
)
