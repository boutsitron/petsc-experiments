"""Experimenting with SLEPc orthogonalization"""

import os

import firedrake as fd
import gmsh
from firedrake import COMM_WORLD
from mpi4py import MPI

indir = "MixedFunctionSpace"
if not os.path.exists(indir):
    os.makedirs(indir)

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size
tab = " " * 4


def set_values_on_subfunctions(PPhi, up_vec, local_dofs_V, local_dofs_Q):
    """Sets values on the subfunctions of a mixed function space

    Args:
        PPhi (fd.Function): Mixed function space
        up_vec (PETSc.Vec): Vector to be passed to mixed function
        local_dofs_V (int): Local number of degrees of freedom for V
        local_dofs_Q (int): Local number of degrees of freedom for Q
    """
    Phiu, Phip = PPhi.subfunctions
    Phiu.rename("Phi_u")
    Phip.rename("Phi_p")

    W = PPhi.function_space()
    dim = W.mesh().topological_dimension()

    local_nodes_V = local_dofs_V // 2

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


def get_local_dofs(V, Q):
    """Get local number of degrees of freedom for V and Q

    Args:
        V (VectorFunctionSpace): Vector function space
        Q (FunctionSpace): Scalar function space

    Returns:
        int, int: Local number of degrees of freedom for V and Q
    """
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

    return local_dofs_V, local_dofs_Q


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

    local_dofs_V, local_dofs_Q = get_local_dofs(V, Q)

    # Optional step to set values on the subfunctions
    set_values_on_subfunctions(PPhi, up_vec, local_dofs_V, local_dofs_Q)

    return PPhi


def save_mixed_bifunction_space(PPhi):
    """Saves a mixed function space to a file

    Args:
        PPhi (fd.Function): Mixed function space to be saved
    """
    W = PPhi.function_space()

    # Get local number of degrees of freedom for V and Q
    h5name = f"{indir}/PPhi.h5"
    with fd.CheckpointFile(h5name, fd.FILE_CREATE) as afile:
        afile.save_mesh(W.mesh())
        afile.save_function(PPhi)


def generate_gmsh_mesh(meshfile: str, mesh_size: float = 0.1):
    """Generate a Gmsh mesh of a unit square domain

    Args:
        meshfile (str): _description_
        mesh_size (float, optional): _description_. Defaults to 0.1.
    """
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.add("unit_square")

    # Create a unit square domain
    _ = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()

    # Apply mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh to a file
    gmsh.write(meshfile)

    gmsh.finalize()


# ------------------------------------------------------------------------------
# TESTING THE FUNCTION THAT PASSES A VECTOR TO A MIXED FUNCTION SPACE FUNCTION
# ------------------------------------------------------------------------------

# mesh = fd.UnitSquareMesh(10, 10)

meshfile = f"{indir}/unit_square.msh"
generate_gmsh_mesh(meshfile)
mesh = fd.Mesh(meshfile)
mesh.name = "mesh"

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

with up.dat.vec_ro as up_vec:
    PPhi = vec_to_mixed_function(up_vec, W)
    Phiu, Phip = PPhi.subfunctions
    Phiu.rename("Phi_u")
    Phip.rename("Phi_p")

    # Get local number of degrees of freedom for V and Q
    local_dofs_V, local_dofs_Q = get_local_dofs(V, Q)

    ownership_start_W, ownership_end_W = up_vec.getOwnershipRange()
    local_dofs_W = ownership_end_W - ownership_start_W

    up_array = up_vec.getArray(readonly=True)
    assert (
        len(up_array) == local_dofs_W
    ), f"The size of up_array {len(up_array)} does not match the global degrees of freedom for W {local_dofs_W}."

    with Phiu.dat.vec_ro as Phiu_vec, Phip.dat.vec_ro as Phip_vec:
        Phiu_array = Phiu_vec.getArray(readonly=True)
        Phip_array = Phip_vec.getArray(readonly=True)
        assert (
            len(Phiu_array) == local_dofs_V
        ), f"The size of Phiu_array {len(Phiu_array)} does not match the global degrees of freedom for V {local_dofs_V}."
        assert (
            len(Phip_array) == local_dofs_Q
        ), f"The size of Phip_array {len(Phip_array)} does not match the global degrees of freedom for Q {local_dofs_Q}."


save_mixed_bifunction_space(PPhi)


phi_pvd = fd.File(f"{indir}/saved_phi.pvd")
phi_pvd.write(Phiu, Phip, u, p)
