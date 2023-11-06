"""Experimenting with SLEPc orthogonalization"""

import contextlib
import sys
import time

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD
from mpi4py import MPI

from utilities import (
    Print,
    create_petsc_diagonal_matrix_seq,
    create_petsc_matrix,
    create_petsc_matrix_seq,
    create_petsc_vector_seq,
    print_matrix_partitioning,
)

with contextlib.suppress(ImportError):
    import slepc4py

    slepc4py.init(sys.argv)
    from slepc4py import SLEPc

from numpy.testing import assert_array_almost_equal

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size
tab = " " * 4


def SVD_slepc(QQ, prnt="off"):
    """
    SVD in PETSc implementation:
    a. performing SVD on Q
    b. taking the left and right singular vectors

    Q = U * S * V.T
    [mxn] = [mxn] * [nxn] * [nxn]

    q  q  q     u  u  u | 0  0     s  s  s
    q  q  q     u  u  u | 0  0     s  s  s     v  v  v
    q  q  q  =  u  u  u | 0  0  *  s  s  s  *  v  v  v
    q  q  q     u  u  u | 0  0     0  0  0     v  v  v
    q  q  q     u  u  u | 0  0     0  0  0
    """
    SVDtime_start = time.time()

    SVD = SLEPc.SVD()
    SVD.create()
    SVD.setOperator(QQ)
    SVD.setType(
        SVD.Type.LAPACK
    )  # CROSS, CYCLIC, LANCZOS, TRLANCZOS, LAPACK, RANDOMIZED, SCALAPACK
    SVD.setFromOptions()
    SVD.solve()

    # kp1 is k+1
    kp1, _ = QQ.getSize()

    # Phin matrix for new SVD: Q = Un * Sn * Vn.T
    PPhin = create_petsc_matrix_seq(np.zeros((kp1, kp1)))  # [k+1 x k+1]

    # Psin matrix for new SVD: Q = Un * Sn * Vn.T
    # PPsin = create_petsc_matrix_seq(np.zeros((kp1, kp1)))  # [k+1 x k+1]

    # Vector to take singular values
    Sn = create_petsc_vector_seq(np.zeros(kp1))

    nconv = SVD.getConverged()
    if nconv > 0:
        v, u = QQ.createVecs()

        for i in range(nconv):
            sigma = SVD.getSingularTriplet(i, u, v)
            error = SVD.computeError(i)
            if prnt == "on":
                Print(f"     sigma = {sigma:6.2e}, error = {error: 12g}")

            Sn.setValues(i, sigma)
            PPhin.setValues(range(kp1), i, u)
            # PPsin.setValues(i, range(kp1), v)

        v.destroy()
        u.destroy()

    SVD.destroy()

    # ------------------------------------------
    PPhin.assemblyBegin()
    PPhin.assemblyEnd()

    # Compute PPhin.T * PPhin
    result_matrix = PPhin.transposeMatMult(PPhin)
    # Create an identity matrix of the same size
    identity_matrix = create_petsc_matrix_seq(np.eye(kp1))
    # Subtract the identity matrix from the result to see if it's close to zero
    result_matrix.axpy(-1, identity_matrix)
    # Compute the Frobenius norm of the resulting matrix
    norm = result_matrix.norm()
    Print(f"    Frobenius norm of PPhin.T * PPhin: {norm:1.2e}")
    # Check if the norm is close to zero within some tolerance
    # tolerance = 1e-13
    # assert (
    #     norm < tolerance
    # ), f"PPhin is not orthonormal, Frobenius norm: {norm:1.2e} > {tolerance:1.2e}"

    SS = create_petsc_diagonal_matrix_seq(Sn)

    SVDtime = time.time() - SVDtime_start
    SVDtime_avg = COMM_WORLD.allreduce(SVDtime, op=MPI.SUM) / nproc
    Print(
        f"{Fore.GREEN}  2.2 SVD of [{kp1:d}x{kp1:d}]: {SVDtime_avg:2.2f} s{Fore.RESET}"
    )
    Sn.destroy()

    return PPhin, SS


# --------------------------------------------
# EXP: Orthogonalization of an mpi PETSc matrix
# --------------------------------------------

m, k = 700, 3
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.rand(m, k)

A = create_petsc_matrix(A_np, sparse=False)
print_matrix_partitioning(A, "A")

# --------------------------------------------------
PPhin, SS = SVD_slepc(A, prnt="off")
print_matrix_partitioning(PPhin, "PPhin", values=False)

# --------------------------------------------
# TEST: Orthogonalization of a numpy matrix
# --------------------------------------------
# Generate A_np_orthogonal
Phin_np, _, _ = np.linalg.svd(A_np, full_matrices=False)  # Perform SVD

# Get the local values from A_orthogonal
local_rows_start, local_rows_end = PPhin.getOwnershipRange()
PPhin_local = PPhin.getValues(range(local_rows_start, local_rows_end), range(k))

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(
    np.abs(PPhin_local),
    np.abs(Phin_np[local_rows_start:local_rows_end, :]),
    decimal=5,
)
