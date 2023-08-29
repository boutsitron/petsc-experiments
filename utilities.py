"""Utility functions for parallel handling"""

from colorama import Fore, Style
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc

rank = COMM_WORLD.rank


def Print(message: str, color: str = Fore.WHITE):
    """Print function that prints only on rank 0 with color

    Args:
        message (str): message to be printed
        color (str, optional): color of the message. Defaults to Fore.WHITE.
    """
    PETSc.Sys.Print(color + message + Style.RESET_ALL)


def print_matrix_partitioning(mat, name="", values=False):  # sourcery skip: move-assign
    """Prints partitioning information of a PETSc MPI matrix.

    Args:
        mat (PETSc Mat): The PETSc MPI matrix.
        name (str): Optional name for the matrix for better identification in printout.
        values (bool): Toggle for printing the local values of the matrix. Defaults to False.
    """
    # Get the local ownership range for rows
    local_rows_start, local_rows_end = mat.getOwnershipRange()
    # Collect all the local ownership ranges and local rows in the root process
    ownership_ranges_rows = COMM_WORLD.gather(
        (local_rows_start, local_rows_end), root=0
    )

    # Initialize an empty list to hold local row values
    local_rows = []
    for i in range(local_rows_start, local_rows_end):
        cols, row_data = mat.getRow(i)
        local_rows.append((i, list(zip(cols, row_data))))
    all_local_rows = COMM_WORLD.gather(local_rows, root=0)

    if rank == 0:
        print(f"MATRIX {name} [{mat.getSize()[0]}x{mat.getSize()[1]}]")
        print(mat.getType())
        print("")
        print(f"Partitioning for {name}:")
        for i, ((start, end), local_rows) in enumerate(
            zip(ownership_ranges_rows, all_local_rows)
        ):
            print(f"  Rank {i}: Rows [{start}, {end})")
            if values:
                for row_idx, row_data in local_rows:
                    print(f"    Row {row_idx}: {row_data}")
        print()


def print_vector_partitioning(vec, name="", values=False):
    """Prints partitioning information and local values of a PETSc MPI vector.

    Args:
        vec (PETSc Vec): The PETSc MPI vector.
        name (str): Optional name for the vector for better identification in printout.
        values (bool): Toggle for printing the local values of the vector. Defaults to False.
    """
    # Get the local ownership range
    local_start, local_end = vec.getOwnershipRange()

    # Get local values for the vector
    local_values = vec.getValues(range(local_start, local_end))

    # Collect all the local ownership ranges and local values in the root process
    ownership_ranges = COMM_WORLD.gather((local_start, local_end), root=0)
    all_local_values = COMM_WORLD.gather(local_values, root=0)

    if rank == 0:
        print(f"VECTOR {name} [{vec.getSize()}x1]")
        print(vec.getType())
        # vec.view()
        print("")
        print(f"Partitioning for {name}:")
        for i, ((start, end), local_vals) in enumerate(
            zip(ownership_ranges, all_local_values)
        ):
            print(f"  Rank {i}: [{start}, {end})")
            if values:
                print(f"  Local Values: {local_vals}")
        print()


def create_petsc_matrix(input_array, partition_like=None, sparse=True):
    """Create a PETSc matrix from an input_array

    Args:
        input_array (np array): Input array
        partition_like (PETSc mat, optional): Petsc matrix. Defaults to None.
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        PETSc mat: PETSc matrix
    """
    # Check if input_array is 1D and reshape if necessary
    assert len(input_array.shape) == 2, "Input array should be 2-dimensional"
    global_rows, global_cols = input_array.shape

    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        # No parallelization in the columns, set local_cols = None to parallelize
        size = ((local_rows, global_rows), (global_cols, global_cols))
    else:
        size = ((None, global_rows), (global_cols, global_cols))

    # Create a sparse or dense matrix based on the 'sparse' argument
    if sparse:
        matrix = PETSc.Mat().createAIJ(size=size, comm=COMM_WORLD)
    else:
        matrix = PETSc.Mat().createDense(size=size, comm=COMM_WORLD)
    matrix.setUp()

    local_rows_start, local_rows_end = matrix.getOwnershipRange()

    for counter, i in enumerate(range(local_rows_start, local_rows_end)):
        # Calculate the correct row in the array for the current process
        row_in_array = counter + local_rows_start
        matrix.setValues(
            i, range(global_cols), input_array[row_in_array, :], addv=False
        )

    # Assembly the matrix to compute the final structure
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


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
    vector = PETSc.Vec().createSeq(size=k, comm=COMM_SELF)

    # Set the values
    vector.setValues(range(k), input_array)

    # Assembly the vector to compute the final structure
    vector.assemblyBegin()
    vector.assemblyEnd()

    return vector


def get_local_submatrix(A):
    """Get the local submatrix of A

    Args:
        A (mpi PETSc mat): partitioned PETSc matrix

    Returns:
        seq mat: PETSc matrix
    """
    local_rows_start, local_rows_end = A.getOwnershipRange()
    local_rows = local_rows_end - local_rows_start
    comm = A.getComm()
    rows = PETSc.IS().createStride(
        local_rows, first=local_rows_start, step=1, comm=comm
    )
    _, k = A.getSize()  # Get the number of columns (k) from A's size
    cols = PETSc.IS().createStride(k, first=0, step=1, comm=comm)

    # print(f"For proc {rank} rows indices: {rows.getIndices()}")
    # Print(f"For proc {rank} cols indices: {cols.getIndices()}")

    # Getting the local submatrix
    # TODO: To be replaced by MatMPIAIJGetLocalMat() in the future (see petsc-users mailing list). There is a missing petsc4py binding, need to add it myself (and please create a merge request)
    A_local = A.createSubMatrices(rows, cols)[0]
    return A_local
