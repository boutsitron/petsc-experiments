"""Experimenting with SLEPc orthogonalization"""

import os

import firedrake as fd
import numpy as np
from firedrake import COMM_WORLD
from mpi4py import MPI

from utilities import create_petsc_vector

indir = "MixedFunctionSpace"
if not os.path.exists(indir):
    os.makedirs(indir)

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size
tab = " " * 4


def vec_to_mixed_function(up_vec, W):
    """Passes a vector to a mixed function

    Args:
        up_vec (PETSc.Vec): Vector to be passed to mixed function
        W (fd.FunctionSpace): Mixed function space

    Returns:
        fd.Function: Mixed function with vector passed to it
    """
    PPhi = fd.Function(W, name="PPhi")
    V, Q = W.subfunctions

    # Get local number of degrees of freedom for V and Q
    mesh = V.mesh()
    dim = mesh.topological_dimension()

    local_nodes_V = V.dof_dset.size
    local_dofs_V = dim * local_nodes_V
    local_dofs_Q = Q.dof_dset.size
    global_dofs_W = W.dim()

    global_dofs_V = COMM_WORLD.allreduce(local_dofs_V, op=MPI.SUM)
    global_dofs_Q = COMM_WORLD.allreduce(local_dofs_Q, op=MPI.SUM)

    assert global_dofs_W == global_dofs_V + global_dofs_Q

    print(
        f"For proc {rank}: local_dofs_V = {local_dofs_V} + local_dofs_Q = {local_dofs_Q} = local_nodes_W = {global_dofs_W}"
    )
    print()

    Phiu, Phip = PPhi.subfunctions
    Phiu.rename("Phi_u")
    Phip.rename("Phi_p")

    # In parallel, each process will only handle its part of the vector.
    with Phiu.dat.vec_wo as Phiu_vec, Phip.dat.vec_wo as Phip_vec:
        # Get the range for this process for V and Q
        ownership_start_V, ownership_end_V = Phiu_vec.getOwnershipRange()
        ownership_start_Q, ownership_end_Q = Phip_vec.getOwnershipRange()

        # Ensure the ownership ranges are consistent with the local degrees of freedom
        assert ownership_end_V - ownership_start_V == local_dofs_V
        assert ownership_end_Q - ownership_start_Q == local_dofs_Q

        # The local portion of the vector that this process is responsible for
        local_up_vec = up_vec.getArray(readonly=True)

        # Now, only assign the values that this process owns
        Phiu_vec.setValues(
            range(ownership_start_V, ownership_end_V),
            local_up_vec[:local_dofs_V].reshape((local_nodes_V, dim)),
        )

        Phip_vec.setValues(
            range(ownership_start_Q, ownership_end_Q),
            local_up_vec[local_dofs_V:],
        )

        # Finalize the insertion of values
        Phiu_vec.assemble()
        Phip_vec.assemble()

    return PPhi


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
mesh = PPhi.function_space().mesh()
Phiu, Phip = PPhi.subfunctions
Phiu.rename("Phi_u")
Phip.rename("Phi_p")

V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)
W = V * Q

up = fd.Function(W, name="solution")

# Initialize up with some expressions for velocity and pressure
x, y = fd.SpatialCoordinate(mesh)
# A complex velocity field example:
# Swirling pattern that increases in magnitude towards the center of the domain
velocity_init = fd.as_vector(
    [fd.sin(fd.pi * x) * fd.cos(fd.pi * y), fd.sin(fd.pi * y) * fd.cos(fd.pi * x)]
)
pressure_init = fd.sin(2 * fd.pi * x) * fd.cos(2 * fd.pi * y)  # Example pressure field

u, p = up.subfunctions
u.interpolate(velocity_init)
p.interpolate(pressure_init)

# Ensure subfunctions have correct names
u.rename("velocity")
p.rename("pressure")

global_rows = W.dim()

with up.dat.vec_ro as up_vec:
    phi_vec = create_petsc_vector(np.zeros(global_rows), partition_like=up_vec)
    assert (
        up_vec.getOwnershipRange() == phi_vec.getOwnershipRange()
    ), f"Ownership range mismatch: up_vec = {up_vec.getOwnershipRange()}, phi_vec = {phi_vec.getOwnershipRange()}"

phi_pvd = fd.File(f"{indir}/loaded_phi.pvd")
phi_pvd.write(Phiu, Phip, u, p)
