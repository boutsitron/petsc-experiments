"""Experimenting with SLEPc orthogonalization"""

import os

import firedrake as fd
from firedrake import COMM_WORLD
from firedrake.output import VTKFile

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

    return mesh, PPhi


# ------------------------------------------------------------------------------
# TESTING THE FUNCTION THAT PASSES A VECTOR TO A MIXED FUNCTION SPACE FUNCTION
# ------------------------------------------------------------------------------


meshfile = f"{indir}/unit_square.msh"
mesh = fd.Mesh(meshfile)
mesh.name = "mesh"
_, PPhi = load_mixed_bifunction_space()

# mesh, PPhi = load_mixed_bifunction_space()

V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)
W = V * Q

up = fd.Function(W, name="solution")
global_rows = W.dim()

with up.dat.vec_ro as up_vec, PPhi.dat.vec_ro as PPhi_vec:

    assert (
        up_vec.getOwnershipRange() == PPhi_vec.getOwnershipRange()
    ), f"Ownership range mismatch: up_vec = {up_vec.getOwnershipRange()}, phi_vec = {PPhi_vec.getOwnershipRange()}"

Phiu, Phip = PPhi.subfunctions
Phiu.rename("Phi_u")
Phip.rename("Phi_p")
phi_pvd = VTKFile(f"{indir}/loaded_phi.pvd")
phi_pvd.write(Phiu, Phip)
