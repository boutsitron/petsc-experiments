"""Experimenting with SLEPc orthogonalization"""

import contextlib
import sys
import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD

from utilities import Print, create_petsc_matrix, print_matrix_partitioning

with contextlib.suppress(ImportError):
    import slepc4py

    slepc4py.init(sys.argv)
    from slepc4py import SLEPc

from numpy.testing import assert_array_almost_equal

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank
EPSILON_SVD = 1e-9
EPS = 1e-12


def orthogonality(PPhi):  # sourcery skip: avoid-builtin-shadow
    """Checking and correcting orthogonality

    Args:
        PPhi (PETSc.Mat): Matrix of size [m x k+1].

    Returns:
        PETSc.Mat: Matrix of size [m x k+1].
    """
    # checking orthogonality
    orth_start = time.time()

    # Check if the matrix is dense
    mat_type = PPhi.getType()
    assert mat_type in (
        "seqdense",
        "mpidense",
    ), "PPhi must be a dense matrix. SLEPc.BV().createFromMat() requires a dense matrix."

    m, kp1 = PPhi.getSize()

    Phi1 = PPhi.getColumnVector(0)
    Phi2 = PPhi.getColumnVector(kp1 - 1)

    # Compute dot product using PETSc function
    dot_product = Phi1.dot(Phi2)

    if abs(dot_product) > min(EPSILON_SVD, EPS * m):
        Print("    Basis has lost (numerical) orthogonality", Fore.RED)

        # Type can be CHOL, GS, mro(), SVQB, TSQR, TSQRCHOL
        _type = SLEPc.BV().OrthogBlockType.GS

        bv = SLEPc.BV().createFromMat(PPhi)
        bv.setFromOptions()
        bv.setOrthogonalization(_type)
        bv.orthogonalize()

        PPhi = bv.createMat()
        print_matrix_partitioning(PPhi, "PPhi after orthogonalization")

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
Q, R = np.linalg.qr(A_np)
A_np_orthogonal = Q

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
