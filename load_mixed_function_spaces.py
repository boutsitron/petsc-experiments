"""Experimenting with SLEPc orthogonalization"""

import os

import firedrake as fd
from firedrake import COMM_WORLD
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

indir = "MixedFunctionSpace"
if not os.path.exists(indir):
    os.makedirs(indir)

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size
tab = " " * 4


def load_mixed_bifunction_space(mesh_name="mesh"):
    """Loads a mixed bifunction space from a file

    Args:
        mesh_name (str, optional): Name of the mesh. Defaults to "mesh".

    Returns:
        fd.Function: Mixed bifunction space
    """
    h5name = f"{indir}/PPhi.h5"
    with fd.CheckpointFile(h5name, fd.FILE_READ) as afile:
        mesh = afile.load_mesh(mesh_name)
        PPhi = afile.load_function(mesh, "PPhi")

    return PPhi


# ------------------------------------------------------------------------------
# TESTING THE FUNCTION THAT PASSES A VECTOR TO A MIXED FUNCTION SPACE FUNCTION
# ------------------------------------------------------------------------------

PPhi = load_mixed_bifunction_space()

W = PPhi.function_space()
up = fd.Function(W, name="solution")
global_rows = W.dim()

with up.dat.vec_ro as up_vec, PPhi.dat.vec_ro as PPhi_vec:

    # Determine the local portion of the vector
    local_start, local_end = up_vec.getOwnershipRange()
    local_size = local_end - local_start

    mat_size = ((local_size, global_rows), (None, 1))

    phi_mat = PETSc.Mat().createAIJ(size=mat_size, comm=COMM_WORLD)
    phi_mat.setUp()

    assert (
        up_vec.getOwnershipRange() == PPhi_vec.getOwnershipRange()
    ), f"Ownership range mismatch: up_vec = {up_vec.getOwnershipRange()}, phi_vec = {PPhi_vec.getOwnershipRange()}"

Phiu, Phip = PPhi.subfunctions
Phiu.rename("Phi_u")
Phip.rename("Phi_p")
phi_pvd = VTKFile(f"{indir}/loaded_phi.pvd")
phi_pvd.write(Phiu, Phip)
