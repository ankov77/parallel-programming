#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[])
{
    double startTime = 0, endTime = 0;
    int taskLength = 0;
    int size = 125;
    int procNum, procRank;
    int min, i, j;
    int **matrix;
    int *minsInColumns;
    MPI_Status status;

    matrix = (int **)malloc(sizeof(int *) * size);
    for (i = 0; i < size; i++)
    {
        matrix[i] = (int *)malloc(sizeof(int) * size);
        for (j = 0; j < size; j++)
        {
            matrix[i][j] = rand() % 200 - 50; // [-50, 150)
        }
    }
    minsInColumns = (int *)malloc(sizeof(int) * size);

    // MPI START
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    taskLength = size / procNum;
    if (size % procNum != 0)
    {
        taskLength++;
    }
    if (procRank == 0)
    {
        startTime = MPI_Wtime();
    }

    for (int i = taskLength * procRank; i < taskLength * (procRank + 1); i++)
    {
        min = matrix[0][i];
        for (int j = 0; j < size; j++)
        {
            if (matrix[j][i] < min)
            {
                min = matrix[j][i];
            }
        }
        if (procRank == 0)
        {
            minsInColumns[i] = min;
            //ParallelMinVector[i] = min;
        }
        else
        {
            MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (procRank == 0)
    {
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix, size * size, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 1; i < procNum; i++)
        {
            for (int j = 0; j < taskLength; j++)
            {
                MPI_Recv(&min, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                minsInColumns[i * taskLength + j] = min;
            }
        }
        endTime = MPI_Wtime(); //END TIME
        printf("Parallel time: %f\n", endTime - startTime);

        startTime = MPI_Wtime();
        for (int i = 0; i < size; i++)
        {
            minsInColumns[i] = matrix[0][i];
            for (int j = 0; j < size; j++)
            {
                if (matrix[j][i] < minsInColumns[i])
                {
                    minsInColumns[i] = matrix[j][i];
                }
            }
        }
        endTime = MPI_Wtime();
        printf("Serial time: %f\n", endTime - startTime);
    }

    MPI_Finalize();
    //MPI END

    for (i = 0; i < size; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
    free(minsInColumns);

    return 0;
}
