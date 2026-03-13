#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1024   // Matrix size (1024x1024)

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;

    // Allocate memory
    int *A = NULL;
    int *AT = NULL;
    int *subA = (int *)malloc(rows_per_proc * N * sizeof(int));
    int *subAT = (int *)malloc(rows_per_proc * N * sizeof(int));

    if (rank == 0)
    {
        A = (int *)malloc(N * N * sizeof(int));
        AT = (int *)malloc(N * N * sizeof(int));

        // Initialize matrix
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i * N + j] = i * N + j;
    }

    // Scatter rows
    MPI_Scatter(A, rows_per_proc * N, MPI_INT,
                subA, rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    // Local transpose
    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N; j++)
            subAT[j * rows_per_proc + i] = subA[i * N + j];

    // Gather transposed blocks
    MPI_Gather(subAT, rows_per_proc * N, MPI_INT,
               AT, rows_per_proc * N, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Original A[0][1] = %d\n", A[0 * N + 1]);
        printf("Transpose AT[1][0] = %d\n", AT[1 * N + 0]);
    }

    free(subA);
    free(subAT);
    if (rank == 0)
    {
        free(A);
        free(AT);
    }

    MPI_Finalize();
    return 0;
}
