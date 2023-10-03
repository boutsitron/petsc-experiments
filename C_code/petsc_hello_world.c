#include <petscsys.h>

const char help[] = "Hello World example program in PETSc.\n\n";

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt rank, size;

  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
  CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Hello world from processor %D of %D\n", rank, size);
  CHKERRQ(ierr);

  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return ierr;
}
