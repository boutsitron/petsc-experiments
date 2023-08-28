"""Utility functions for parallel handling"""

from firedrake import COMM_WORLD
from firedrake.petsc import PETSc

rank = COMM_WORLD.rank


def Print(x: str):
    """Prints the string only on the root process

    Args:
        x (str): String to be printed
    """
    PETSc.Sys.Print(x)


def print_mat_info(mat, name):
    """Prints the matrix information

    Args:
        mat (PETSc mat): PETSc matrix
        name (string): Name of the matrix
    """
    Print(f"MATRIX {name} [{mat.getSize()[0]}x{mat.getSize()[1]}]")
    # print(f"For rank {rank} local {name}: {mat.getSizes()}")
    Print(mat.getType())
    mat.view()
    COMM_WORLD.Barrier()
    Print("")


def print_vector_partitioning(vec, name=""):
    """Prints partitioning information and local values of a PETSc MPI vector.

    Args:
        vec (PETSc Vec): The PETSc MPI vector.
        name (str): Optional name for the vector for better identification in printout.
    """
    # Get the local ownership range
    local_start, local_end = vec.getOwnershipRange()

    # Get local values for the vector
    local_values = vec.getValues(range(local_start, local_end))

    # Collect all the local ownership ranges and local values in the root process
    ownership_ranges = COMM_WORLD.gather((local_start, local_end), root=0)
    all_local_values = COMM_WORLD.gather(local_values, root=0)

    if COMM_WORLD.rank == 0:
        print(f"Partitioning and local values for {name} vector:")
        for i, ((start, end), local_vals) in enumerate(
            zip(ownership_ranges, all_local_values)
        ):
            print(f"  Rank {i}: [{start}, {end})")
            print(f"  Local Values: {local_vals}")
        print()


def print_matrix_partitioning(mat, name=""):
    """Prints partitioning information of a PETSc MPI matrix.

    Args:
        mat (PETSc Mat): The PETSc MPI matrix.
        name (str): Optional name for the matrix for better identification in printout.
    """
    # Get the local ownership range for rows
    local_rows_start, local_rows_end = mat.getOwnershipRange()

    # Collect all the local ownership ranges in the root process
    ownership_ranges_rows = COMM_WORLD.gather(
        (local_rows_start, local_rows_end), root=0
    )

    if rank == 0:
        print(f"Partitioning for {name} matrix:")
        print("  Rows:")
        for i, (start, end) in enumerate(ownership_ranges_rows):
            print(f"    Rank {i}: [{start}, {end})")
        print()


def create_petsc_vector_seq(input_array):
    """Create a PETSc sequential vector from an input array

    Args:
        input_array (np array): Input 1-dimensional array

    Returns:
        PETSc Vec: PETSc sequential vector
    """
    # Check if input_array is 1D and reshape if necessary
    if len(input_array.shape) != 1:
        raise ValueError("Input array should be 1-dimensional")

    k = input_array.shape[0]

    # Create a sequential vector
    vector = PETSc.Vec().createSeq(size=k, comm=PETSc.COMM_SELF)

    # Set the values
    vector.setValues(range(k), input_array)

    # Assembly the vector to compute the final structure
    vector.assemblyBegin()
    vector.assemblyEnd()

    return vector
