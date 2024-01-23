"""Experimenting with SLEPc SVD"""

import numpy as np
from firedrake import COMM_WORLD
from numpy.testing import assert_array_almost_equal

from utilities import (
    SVD_slepc,
    check_orthonormality,
    convert_seq_matrix_to_global,
    create_petsc_matrix_seq,
    print_matrix_partitioning,
)

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size
tab = " " * 4


# --------------------------------------------
# EXP: Orthogonalization of an mpi PETSc matrix
# --------------------------------------------

m, k = 2700, 10
# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.rand(m, k)

A = create_petsc_matrix_seq(A_np)
print_matrix_partitioning(A, "A")

# --------------------------------------------------
PPhi_seq, SS = SVD_slepc(A, prnt="on")
print_matrix_partitioning(PPhi_seq, "PPhi_seq", values=False)
check_orthonormality(PPhi_seq)

PPhi = convert_seq_matrix_to_global(PPhi_seq)
check_orthonormality(PPhi)

# --------------------------------------------
# TEST: Orthogonalization of a numpy matrix
# --------------------------------------------
# Generate A_np_orthogonal
Phi_np, _, _ = np.linalg.svd(A_np, full_matrices=False)  # Perform SVD

# Get the local values from A_orthogonal
local_rows_start, local_rows_end = PPhi.getOwnershipRange()
PPhi_local = PPhi.getValues(range(local_rows_start, local_rows_end), range(k))

# Assert the correctness of the multiplication for the local subset
assert_array_almost_equal(
    np.abs(PPhi_local),
    np.abs(Phi_np[local_rows_start:local_rows_end, :]),
    decimal=5,
)
