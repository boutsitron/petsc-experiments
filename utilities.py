"""Utility functions for parallel handling"""

import contextlib
import sys
import time

import numpy as np
from colorama import Fore, Style
from firedrake import COMM_SELF, COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI

with contextlib.suppress(ImportError):
    import slepc4py

    slepc4py.init(sys.argv)
    from slepc4py import SLEPc

rank = COMM_WORLD.rank
nproc = COMM_WORLD.size

# --------------------------------------------
# Parallel print functions
# --------------------------------------------


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
        print(
            f"{Fore.YELLOW}MATRIX {mat.getType()} {name} [{mat.getSize()[0]}x{mat.getSize()[1]}]{Style.RESET_ALL}"
        )
        if mat.isAssembled():
            print(f"{Fore.GREEN}Assembled{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Not Assembled{Style.RESET_ALL}")
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
        print(
            f"{Fore.YELLOW}VECTOR {name} {vec.getType()} [{vec.getSize()}x1]{Fore.RESET}"
        )
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


# --------------------------------------------
# PETSc vectors
# --------------------------------------------


def create_petsc_vector(input_array, partition_like=None):
    """Create a PETSc vector from an input_array

    Args:
        input_array (np array): Input array
        partition_like (PETSc mat, optional): Petsc matrix. Defaults to None.

    Returns:
        PETSc vec: PETSc vector
    """
    # Check if input_array is 1D and reshape if necessary
    if len(input_array.shape) != 1:
        raise ValueError("Input array should be 1-dimensional")

    global_size = input_array.shape[0]

    # Determine the local portion of the vector
    if partition_like is not None:
        local_start, local_end = partition_like.getOwnershipRange()
        local_size = local_end - local_start

        size = (local_size, global_size)
        vector = PETSc.Vec().createMPI(size, comm=COMM_WORLD)

    else:
        vector = PETSc.Vec().createMPI(global_size, comm=COMM_WORLD)
        local_start, local_end = vector.getOwnershipRange()

    vector.setUp()

    # Assign the values to the vector
    for counter, i in enumerate(range(local_start, local_end)):
        # Calculate the correct row in the array for the current process
        i_in_array = counter + local_start
        vector.setValues(i, input_array[i_in_array], addv=False)

    # Assembly the vector to compute the final structure
    vector.assemblyBegin()
    vector.assemblyEnd()

    return vector


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


def get_local_subvector(b):
    """Get the local subvector of a PETSc global vector b.

    Args:
        b (mpi PETSc Vec): partitioned PETSc vector

    Returns:
        seq Vec: PETSc vector containing only the locally owned entries
    """
    # Get the local ownership range
    local_start, local_end = b.getOwnershipRange()

    # Create an index set for the local range
    comm = b.getComm()
    local_indices = PETSc.IS().createStride(
        local_end - local_start, first=local_start, step=1, comm=comm
    )

    # Create a sequential vector to hold the local subvector
    local_size = local_end - local_start
    b_local = PETSc.Vec().createSeq(local_size, comm=COMM_SELF)

    # Extract the local subvector
    b.getValues(local_indices, b_local)

    # Clean up
    local_indices.destroy()

    return b_local


# --------------------------------------------
# PETSc matrices
# --------------------------------------------


def create_petsc_matrix(input_array, partition_like=None, sparse=True):
    """Create a PETSc matrix from an input_array

    Args:
        input_array (np array): Input array
        partition_like (PETSc mat, optional): Petsc matrix. Defaults to None.
        sparse (bool, optional): Toggle for sparese or dense. Defaults to True.

    Returns:
        PETSc mat: PETSc mpi matrix
    """
    # Check if input_array is 1D and reshape if necessary
    assert len(input_array.shape) == 2, "Input array should be 2-dimensional"
    global_rows, global_cols = input_array.shape

    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        # No parallelization in the columns, set local_cols = None to parallelize
        size = ((local_rows, global_rows), (None, global_cols))
    else:
        size = ((None, global_rows), (None, global_cols))

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


def create_petsc_matrix_seq(input_array):
    """Building a sequential PETSc matrix from an array

    Args:
        input_array (np array): Input array

    Returns:
        seq mat: PETSc matrix
    """
    assert len(input_array.shape) == 2

    m, n = input_array.shape
    matrix = PETSc.Mat().createAIJ(size=(m, n), comm=COMM_SELF)
    matrix.setUp()

    matrix.setValues(range(m), range(n), input_array, addv=False)

    # Assembly the matrix to compute the final structure
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


def create_petsc_diagonal_matrix_seq(vector):
    """Building a PETSc diagonal matrix from a vector

    Args:
        vector (PETSc Vec): PETSc vector

    Returns:
        seqaij: PETSc diagonal matrix
    """
    k = vector.getSize()
    # Create a sparse or dense matrix based on the 'sparse' argument
    matrix = PETSc.Mat().createAIJ(size=(k, k), comm=COMM_SELF)
    matrix.setUp()

    matrix.setDiagonal(vector)

    matrix.assemblyBegin()
    matrix.assemblyEnd()

    return matrix


# --------------------------------------------
# PETSc conversions
# --------------------------------------------


def convert_global_matrix_to_seq(A):
    """Convert a partitioned matrix to a sequential one such that each processor holds a duplicate of the full matrix.

    Args:
        A (PETSc.Mat): The partitioned matrix

    Returns:
        PETSc.Mat: The sequential matrix
    """
    # Step 1: Get the local submatrix
    A_local = get_local_submatrix(A)

    # Step 2: Convert the local submatrix to numpy array
    A_local_rows, A_local_cols = A_local.getSize()
    start, end = A_local.getOwnershipRange()

    A_local_array = np.zeros((A_local_rows, A_local_cols))
    for i in range(start, end):
        cols, values = A_local.getRow(i)
        A_local_array[i, cols] = values

    A_local.destroy()

    # Step 3: Use allgather to collect all local matrices to all processes
    gathered_data = COMM_WORLD.allgather(A_local_array)

    # Step 4: Stack the local matrices to create the full sequential matrix
    full_matrix_array = np.vstack(gathered_data)

    # Step 5: Create a new sequential PETSc matrix
    m, n = full_matrix_array.shape
    if "aij" in A.getType():
        A_seq = PETSc.Mat().createAIJ([m, n], comm=COMM_SELF)
    else:
        A_seq = PETSc.Mat().createDense([m, n], comm=COMM_SELF)
    A_seq.setUp()

    for i in range(m):
        A_seq.setValues(i, list(range(n)), full_matrix_array[i, :])

    A_seq.assemblyBegin()
    A_seq.assemblyEnd()

    return A_seq


def convert_seq_matrix_to_global(A_seq, partition=None):
    """Convert a duplicated sequential matrix to a partitioned global matrix.

    Args:
        A_seq (PETSc.Mat): Sequential matrix that is duplicated across all processors.
        partition (tuple, optional): The partition of the global matrix. Defaults to None.

    Returns:
        PETSc.Mat: A partitioned global matrix.
    """
    global_rows, global_cols = A_seq.getSize()

    # Determine the local portion of the vector
    if partition is not None:
        local_rows_start, local_rows_end = partition
        local_rows = local_rows_end - local_rows_start

        size = ((local_rows, global_rows), (None, global_cols))
    else:
        size = ((None, global_rows), (None, global_cols))

    # Create the global partitioned matrix with the same dimensions
    A_global = PETSc.Mat().createAIJ(size=size, comm=COMM_WORLD)
    A_global.setUp()

    # Determine the rows that this process will own in the global matrix
    local_rows_start, local_rows_end = A_global.getOwnershipRange()

    # Populate the global matrix
    for i in range(local_rows_start, local_rows_end):
        cols, values = A_seq.getRow(i)
        A_global.setValues(i, cols, values)

    A_global.assemblyBegin()
    A_global.assemblyEnd()

    return A_global


# --------------------------------------------
# PETSc matrix operations
# --------------------------------------------


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

    # Getting the local submatrix
    # TODO: To be replaced by MatMPIAIJGetLocalMat() in the future (see https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg46778.html).
    # There is a missing petsc4py binding: https://gitlab.com/petsc/petsc/-/issues/1443
    A_local = A.createSubMatrices(rows, cols)[0]
    return A_local


def concatenate_row_wise(
    local_matrix, global_matrix, local_matrix_rows, global_row_start
):
    """Concatenate the local matrix to the global matrix

    Args:
        local_matrix (int): local submatrix of global_matrix
        global_matrix (PETSc mat): global matrix
        local_matrix_rows (int): number of rows in the local matrix
        global_row_start (int): starting global row for the local matrix

    Returns:
        PETSc mat: global matrix
    """
    for i in range(local_matrix_rows):
        cols, values = local_matrix.getRow(i)
        global_row = i + global_row_start
        global_matrix.setValues(global_row, cols, values)

    return global_matrix


def concatenate_col_wise(
    local_matrix, global_matrix, local_matrix_rows, global_row_start, local_matrix_cols
):
    """Concatenate the local matrix to the global matrix

    Args:
        local_matrix (PETSc mat): local submatrix of global_matrix
        global_matrix (PETSc mat): global matrix
        local_matrix_rows (int): number of rows in the local matrix
        global_row_start (int): starting global row for the local matrix
        local_matrix_cols (int): number of columns in the local matrix

    Returns:
        PETSc mat: global matrix
    """
    all_values = []
    all_global_rows = [i + global_row_start for i in range(local_matrix_rows)]
    all_values = [local_matrix.getRow(i)[1] for i in range(len(all_global_rows))]

    for j in range(local_matrix_cols):
        values = [all_values[i][j] for i in range(len(all_values))]
        global_matrix.setValues(all_global_rows, j, values)

    return global_matrix


def concatenate_local_to_global_matrix(
    local_matrix, partition_like=None, mat_type=None
):
    """Create the global matrix C from the local submatrix local_matrix

    Args:
        local_matrix (seqaij): local submatrix of global_matrix
        partition_like (mpiaij): partitioned PETSc matrix
        mat_type (str): type of the global matrix. Defaults to None. If None, the type of local_matrix is used.

    Returns:
        mpi PETSc mat: partitioned PETSc matrix
    """
    local_matrix_rows, local_matrix_cols = local_matrix.getSize()
    global_rows = COMM_WORLD.allreduce(local_matrix_rows, op=MPI.SUM)

    # Determine the local portion of the vector
    if partition_like is not None:
        local_rows_start, local_rows_end = partition_like.getOwnershipRange()
        local_rows = local_rows_end - local_rows_start

        size = ((local_rows, global_rows), (None, local_matrix_cols))
    else:
        size = ((None, global_rows), (None, local_matrix_cols))

    if mat_type is None:
        mat_type = local_matrix.getType()

    if "dense" in mat_type:
        sparse = False
    else:
        sparse = True

    if sparse:
        global_matrix = PETSc.Mat().createAIJ(size=size, comm=COMM_WORLD)
    else:
        global_matrix = PETSc.Mat().createDense(size=size, comm=COMM_WORLD)
    global_matrix.setUp()

    # The exscan operation is used to get the starting global row for each process.
    # The result of the exclusive scan is the sum of the local rows from previous ranks.
    global_row_start = COMM_WORLD.exscan(local_matrix_rows, op=MPI.SUM)
    if rank == 0:
        global_row_start = 0

    if local_matrix_cols <= local_matrix_rows:
        global_matrix = concatenate_col_wise(
            local_matrix,
            global_matrix,
            local_matrix_rows,
            global_row_start,
            local_matrix_cols,
        )
    else:
        global_matrix = concatenate_row_wise(
            local_matrix, global_matrix, local_matrix_rows, global_row_start
        )

    global_matrix.assemblyBegin()
    global_matrix.assemblyEnd()

    return global_matrix


def SVD_slepc(QQ, prnt="off"):
    """SVD in PETSc implementation:
    a. performing SVD on Q
    b. taking the left and right singular vectors

    Q = U * S * V.T
    [mxn] = [mxn] * [nxn] * [nxn]

    q  q  q     u  u  u | 0  0     s  s  s
    q  q  q     u  u  u | 0  0     s  s  s     v  v  v
    q  q  q  =  u  u  u | 0  0  *  s  s  s  *  v  v  v
    q  q  q     u  u  u | 0  0     0  0  0     v  v  v
    q  q  q     u  u  u | 0  0     0  0  0

    Args:
        QQ (PETSc.Mat): matrix to perform SVD on
        prnt (str, optional): Print option. Defaults to "off".

    Returns:
        PPhin (PETSc.Mat): left singular vectors
        SS (PETSc.Mat): singular values
    """
    SVDtime_start = time.time()

    SVD = SLEPc.SVD()
    # Add this after creating SVD and before SVD.solve()
    SVD.create()
    SVD.setOperator(QQ)
    SVD.setType(
        SVD.Type.LAPACK
    )  # CROSS, CYCLIC, LANCZOS, TRLANCZOS, LAPACK, RANDOMIZED, SCALAPACK
    SVD.setFromOptions()
    SVD.solve()

    # kp1 is k+1
    m, n = QQ.getSize()  # Assuming QQ is m x n matrix
    nconv = SVD.getConverged()

    # Initialize Phin (U) and Vn matrices
    PPhin = create_petsc_matrix_seq(np.zeros((m, nconv)))  # [m x nconv]
    # PPsin = create_petsc_matrix_seq(np.zeros((n, nconv)))  # [n x nconv]

    # Initialize Sn vector to hold singular values
    Sn = create_petsc_vector_seq(np.zeros(nconv))

    if nconv > 0:
        v, u = QQ.createVecs()

        for i in range(nconv):
            sigma = SVD.getSingularTriplet(i, u, v)
            error = SVD.computeError(i)
            if prnt == "on":
                Print(f"     sigma = {sigma:6.2e}, error = {error: 12g}")

            Sn.setValues(i, sigma)
            PPhin.setValues(range(m), i, u)
            # PPsin.setValues(i, range(n), v)

        v.destroy()
        u.destroy()

    SVD.destroy()

    # ------------------------------------------
    PPhin.assemblyBegin()
    PPhin.assemblyEnd()

    # Add this right after you compute PPhin
    # PPhin_values = PPhin.getValues(range(m), range(nconv))
    # Print(f"PPhin matrix values:\n {PPhin_values}")

    SS = create_petsc_diagonal_matrix_seq(Sn)

    SVDtime = time.time() - SVDtime_start
    SVDtime_avg = COMM_WORLD.allreduce(SVDtime, op=MPI.SUM) / nproc
    Print(f"{Fore.GREEN}  2.2 SVD of [{m:d}x{n:d}]: {SVDtime_avg:2.2f} s{Fore.RESET}")
    Sn.destroy()

    return PPhin, SS


def check_orthonormality(PPhi, tolerance=1e-13):
    """Check if the matrix PPhi is orthonormal

    Args:
        PPhi (PETSc.Mat): matrix to check
        tolerance (float, optional): tolerance for the Frobenius norm. Defaults to 1e-13.
    """
    _, k = PPhi.getSize()
    # Compute PPhi.T * PPhi
    result_matrix = PPhi.transposeMatMult(PPhi)

    print_matrix_partitioning(result_matrix, "PPhi.T * PPhi", values=False)
    result_matrix.view()

    # Create an identity matrix of the same size
    if "mpi" in result_matrix.getType():
        identity_matrix = create_petsc_matrix(np.eye(k))
    else:
        identity_matrix = create_petsc_matrix_seq(np.eye(k))
    # Subtract the identity matrix from the result to see if it's close to zero
    result_matrix.axpy(-1, identity_matrix)
    # Compute the Frobenius norm of the resulting matrix
    norm = result_matrix.norm()
    Print(f"    Frobenius norm of PPhi.T * PPhi: {norm:1.2e}")
    # Check if the norm is close to zero within some tolerance
    assert (
        norm < tolerance
    ), f"PPhi is not orthonormal, Frobenius norm: {norm:1.2e} > {tolerance:1.2e}"
