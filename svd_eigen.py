"""Experimenting with SVD using Eigenvalues"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

m, n = 4, 2
A = np.zeros((m, n))
A[::2, 0] = 1
A[1::2, 1] = 1
matrix_a = PETSc.Mat().createDense([m, n], array=A, comm=MPI.COMM_SELF)
matrix_a.assemble()

matrix_ata = matrix_a.transposeMatMult(matrix_a)
matrix_ata.assemble()

eigen_solver = SLEPc.EPS().create(comm=MPI.COMM_SELF)
eigen_solver.setOperators(matrix_ata)
eigen_solver.setDimensions(n)
eigen_solver.setType("lapack")
eigen_solver.setTolerances(1e-6, 100)
eigen_solver.solve()
phi = PETSc.Mat().createDense([m, n], comm=MPI.COMM_SELF)
phi.setUp()
sigma = PETSc.Mat().createDense([n, n], comm=MPI.COMM_SELF)
sigma.setUp()
psiT = PETSc.Mat().createDense([n, n], comm=MPI.COMM_SELF)
psiT.setUp()
for i in range(n):
    vr, vi = matrix_ata.createVecs()
    k = eigen_solver.getEigenpair(i, vr, vi)
    rightvec = psiT.getDenseColumnVec(i, mode="w")
    rightvec.axpy(1.0, vr)
    leftvec = phi.getDenseColumnVec(i, mode="w")
    matrix_a.mult(rightvec, leftvec)
    leftvec.scale(1.0 / np.sqrt(k.real))
    psiT.restoreDenseColumnVec(i, mode="w")
    phi.restoreDenseColumnVec(i, mode="w")
    sigma.setValue(i, i, np.sqrt(k.real))
phi.assemble()
sigma.assemble()
psiT.assemble()

phi_np, sigma_np, psiT_np = np.linalg.svd(A, full_matrices=False)
assert (
    np.linalg.norm(np.abs(phi.getDenseArray()) - np.abs(phi_np)) < 1e-10
), f"norm difference between phi and phi_np is: {np.linalg.norm(np.abs(phi.getDenseArray()) - np.abs(phi_np))}"
assert (
    np.linalg.norm(np.abs(sigma.getDiagonal().array) - np.abs(sigma_np)) < 1e-10
), f"norm difference between sigma and sigma_np is: {np.linalg.norm(np.abs(sigma.getDiagonal().array) - np.abs(sigma_np))}"
assert (
    np.linalg.norm(np.abs(psiT.getDenseArray()) - np.abs(psiT_np.T)) < 1e-10
), f"norm difference between psiT and psiT_np is: {np.linalg.norm(np.abs(psiT.getDenseArray()) - np.abs(psiT_np))}"

# Get just the diagonal values from sigma matrix for comparison
sigma_diag = sigma.getDiagonal().array

assert np.allclose(
    sigma_diag, sigma_np
), f"sigma diagonal: {sigma_diag} is not close to sigma_np: {sigma_np}"
