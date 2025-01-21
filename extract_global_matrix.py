"""Experimenting with PETSc mat-mat multiplication"""

import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD

from utilities import Print, create_petsc_matrix, print_matrix_partitioning

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


# --------------------------------------------
# EXP: Get a subset of a global matrix into another global matrix
#  C = A[0:m, 0:k]
# --------------------------------------------

m, k, l = 7, 5, 3  # noqa: E741

# Generate the random numpy matrices
np.random.seed(0)  # sets the seed to 0
A_np = np.random.randint(low=0, high=6, size=(m, k))

Print(f"My initial marix: {A_np}", Fore.MAGENTA)

A = create_petsc_matrix(A_np)
print_matrix_partitioning(A, "A")

A.view()

A_subset_np = A_np[:, :l]
A_subset = create_petsc_matrix(A_subset_np, partition_like=A)
print_matrix_partitioning(A_subset, "A_subset")

local_matrix_rows, local_matrix_cols = A.getSize()
local_rows_start, local_rows_end = A.getOwnershipRange()

column_vectors = []

# Get the column vector for every column in A
for col in range(local_matrix_cols):
    column_vectors.append(A.getColumnVector(col))


for col in range(l):
    A_subset.setValues(
        range(local_rows_start, local_rows_end), col, column_vectors[col], addv=False
    )

A_subset.assemblyBegin()
A_subset.assemblyEnd()

A_subset.view()

print_matrix_partitioning(A_subset, "A_subset")
