"""Experimenting with SLEPc orthogonalization"""

import contextlib
import sys
import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD

from utilities import (
    Print,
    convert_global_matrix_to_seq,
    convert_seq_matrix_to_global,
    create_petsc_matrix,
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

        local_rows_start, local_rows_end = PPhi.getOwnershipRange()

        # Type can be CHOL, GS, mro(), SVQB, TSQR, TSQRCHOL
        _type = SLEPc.BV().OrthogBlockType.GS

        # Check if the matrix is dense
        mat_type = PPhi.getType()

        # if it's dense it's good to go
        if "dense" in mat_type:
            bv = SLEPc.BV().createFromMat(PPhi)
        # if it's sparse and partitioned, convert it to sequential and then to dense
        elif "seq" not in mat_type:
            PPhi_seq = convert_global_matrix_to_seq(PPhi)
            bv = SLEPc.BV().createFromMat(PPhi_seq.convert("dense"))
        # if it's sparse and sequential, convert it to dense
        else:
            bv = SLEPc.BV().createFromMat(PPhi.convert("dense"))
        bv.setFromOptions()
        bv.setOrthogonalization(_type)
        bv.orthogonalize()

        PPhi = bv.createMat()

        if "seq" in PPhi.getType():
            PPhi = convert_seq_matrix_to_global(
                PPhi, partition=(local_rows_start, local_rows_end)
            )

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
