"""Experimenting with SLEPc orthogonalization"""

import os

import firedrake as fd
from firedrake import COMM_WORLD
from mpi4py import MPI

indir = "POMs"
if not os.path.exists(indir):
    os.makedirs(indir)

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size
tab = " " * 4


def vec_to_mixed_function(up_vec, PPhi):
    """Passes a vector to a mixed function

    Args:
        up_vec (PETSc.Vec): Vector to be passed to mixed function
        PPhi (fd.Function): Mixed function to be passed to

    Returns:
        fd.Function: Mixed function with vector passed to it
    """
    W = up.function_space()
    V, Q = W.split()

    # Get local number of degrees of freedom for V and Q
    local_nodes = V.dof_dset.size
    # local_cells = Q.dof_dset.size
    NODES = COMM_WORLD.allreduce(local_nodes, op=MPI.SUM)
    # NCELLS = COMM_WORLD.allreduce(local_cells, op=MPI.SUM)

    # NDOFS = 2 * NODES + NCELLS

    Phiu, Phip = PPhi.subfunctions
    Phiu.rename("Phi_u")
    Phip.rename("Phi_p")

    TNODES = 2 * NODES

    # converting numpy array back to fd.function
    with Phiu.dat.vec as Phiu_vec, Phip.dat.vec as Phip_vec:
        Phiu_vec[:] = up_vec[:TNODES].reshape((NODES, 2))
        Phip_vec[:] = up_vec[TNODES:]

    return PPhi


mesh = fd.UnitSquareMesh(10, 10)

V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)
W = V * Q

up = fd.Function(W, name="solution")
PPhi = fd.Function(W, name="PPhi")

# Initialize up with some expressions for velocity and pressure
x, y = fd.SpatialCoordinate(mesh)
# A complex velocity field example:
# Swirling pattern that increases in magnitude towards the center of the domain
velocity_init = fd.as_vector(
    [fd.sin(fd.pi * x) * fd.cos(fd.pi * y), fd.sin(fd.pi * y) * fd.cos(fd.pi * x)]
)
pressure_init = fd.sin(2 * fd.pi * x) * fd.cos(2 * fd.pi * y)  # Example pressure field

u, p = up.split()
u.interpolate(velocity_init)
p.interpolate(pressure_init)

# Ensure subfunctions have correct names
u.rename("velocity")
p.rename("pressure")


with up.dat.vec_ro as up_vec:
    PPhi = vec_to_mixed_function(up_vec, PPhi)
    Phiu, Phip = PPhi.subfunctions
    Phiu.rename("Phi_u")
    Phip.rename("Phi_p")

phi_pvd = fd.File(f"{indir}/phi.pvd")

phi_pvd.write(Phiu, Phip, u, p)
