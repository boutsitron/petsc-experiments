"""
This script demonstrates the block preconditioner for the snapshots matrix.

The block preconditioner is a 2x2 block matrix that is used to precondition the snapshots matrix.
"""

from typing import Optional

from firedrake import Function, UnitSquareMesh, VectorFunctionSpace
from mpi4py import MPI
from petsc4py import PETSc


def assert_condition(condition, message):
    """Assert a condition with a custom error message."""
    if not condition:
        raise AssertionError(message)


def build_block_preconditioner(
    A: PETSc.Mat, B: PETSc.Mat, C: Optional[PETSc.Mat] = None
) -> PETSc.Mat:
    r"""
    Build a block preconditioner for the snapshots matrix.

    The structure depends on whether parameter C is provided:

    If C is None, the structure is a 2x2 block matrix:

    .. math::
        \begin{bmatrix}
            A & ZZu \\
            ZZp & B
        \end{bmatrix}

    If C is provided, the upper-left block becomes a nested matrix [A, C]:

    .. math::
        \begin{bmatrix}
            [A, C] & ZZu \\
            ZZp & B
        \end{bmatrix}

    Args:
        A: The first diagonal block for the resulting matrix
        B: The second diagonal block for the resulting matrix
        C: Optional additional block to create a nested matrix with A in the upper-left position

    Returns:
        The block preconditioner for the snapshots matrix
    """
    assert_condition(
        A.getComm() == B.getComm(),
        "A, B, and C must have the same communicator",
    )

    lower_block = B
    if C is not None:
        assert_condition(
            A.getComm() == C.getComm(),
            "A and C must have the same communicator",
        )
        upper_block = PETSc.Mat().createNest([[A, C]], comm=A.getComm())
        upper_block.convert("dense")

        upper_block.assemblyBegin()
        upper_block.assemblyEnd()
    else:
        upper_block = A

    ZZu = PETSc.Mat().createDense(
        size=(
            upper_block.getSizes()[0],
            lower_block.getSizes()[1],
        ),
        comm=A.getComm(),
    )
    ZZu.setUp()

    ZZp = PETSc.Mat().createDense(
        size=(
            lower_block.getSizes()[0],
            upper_block.getSizes()[1],
        ),
        comm=A.getComm(),
    )
    ZZp.setUp()

    basis_matrix = PETSc.Mat().createNest(
        [[upper_block, ZZu], [ZZp, lower_block]], comm=A.getComm()
    )

    basis_matrix.assemblyBegin()
    basis_matrix.assemblyEnd()

    return basis_matrix


def create_random_matrix(size):
    """Create a random matrix of given size."""
    mat = PETSc.Mat().createDense(size=size, comm=MPI.COMM_WORLD)
    mat.setUp()
    mat.setRandom()
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def snapshot_list():
    """Create a list of snapshots from a unit square mesh."""
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 2)

    num_snapshots = 5
    up = Function(V)

    with up.dat.vec_ro as sol_vec:
        snapshots_matrix = PETSc.Mat().createDense(
            size=(
                sol_vec.getSizes(),
                num_snapshots,
            ),
            comm=MPI.COMM_WORLD,
        )
        snapshots_matrix.setUp()
        vec = snapshots_matrix.getDenseColumnVec(0, mode="w")
        vec.axpy(1.0, sol_vec)
        snapshots_matrix.restoreDenseColumnVec(0, mode="w")

    for i in range(1, num_snapshots):
        up.assign(i)
        with up.dat.vec_ro as sol_vec:
            vec = snapshots_matrix.getDenseColumnVec(i, mode="w")
            vec.axpy(1.0, sol_vec)
            snapshots_matrix.restoreDenseColumnVec(i, mode="w")

    return snapshots_matrix


def main():
    """Main function to demonstrate the block preconditioner."""
    # Get MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Creating snapshots matrix...")

    snapshots = snapshot_list()

    if rank == 0:
        print(f"Snapshots matrix size: {snapshots.getSize()}")
        print("\nCreating block preconditioner...")

    A_ = snapshots
    B_ = snapshots

    A_.assemblyBegin()
    A_.assemblyEnd()
    B_.assemblyBegin()
    B_.assemblyEnd()

    if rank == 0:
        print(f"Block A size: {A_.getSize()}")
        print(f"Block B size: {B_.getSize()}")
        use_random = input("Use random matrices instead? (y/n): ").lower() == "y"
        use_c_block = input("Include C block in preconditioner? (y/n): ").lower() == "y"
    else:
        use_random = False
        use_c_block = False

    # Broadcast user choices to all processes
    use_random = comm.bcast(use_random, root=0)
    use_c_block = comm.bcast(use_c_block, root=0)

    if use_random:
        A_ = create_random_matrix((5, 3))
        B_ = create_random_matrix((3, 2))
        C_ = create_random_matrix((5, 2)) if use_c_block else None
        if rank == 0:
            print(f"Random matrix A size: {A_.getSize()}")
            print(f"Random matrix B size: {B_.getSize()}")
            if use_c_block:
                print(f"Random matrix C size: {C_.getSize()}")
    else:
        C_ = None
        if use_c_block:
            if rank == 0:
                print("Using A as C block for demonstration...")
            C_ = A_

    try:
        C = (
            build_block_preconditioner(A_, B_, C_)
            if use_c_block
            else build_block_preconditioner(A_, B_)
        )
        if rank == 0:
            print("\nBlock preconditioner created successfully!")
            print(f"Final matrix size: {C.getSize()}")
            print("\nAttempting to convert to AIJ format...")

        C.convert("aij")

        if rank == 0:
            print("Conversion successful!")
    except Exception as e:
        if rank == 0:
            print(f"\nError occurred: {str(e)}")
            print(
                "\nNote: The conversion to AIJ format might fail with snapshot matrices."
            )
            print("Try using random matrices for a working example.")


if __name__ == "__main__":
    main()
