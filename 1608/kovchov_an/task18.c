#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[])
{
    double startTime = 0, endTime = 0;
    int taskLength = 0;
    int rows, columns;
    int procNum, procRank;
    int min, i, j;
    int *matrix;
    int *minsInColumns;
    MPI_Status status;

    // MPI START
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (procRank == 0)
    {
        if (argc == 3)
        {
            rows = atoi(argv[1]);
            columns = atoi(argv[2]);
        }
        else
        {
            rows = 1000;
            columns = 1000;
        }
        startTime = MPI_Wtime();
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

    matrix = (int *)malloc(sizeof(int) * rows * columns);
    minsInColumns = (int *)malloc(sizeof(int) * columns);

    if (procRank == 0)
    {
        srand(time(NULL));
        for (i = 0; i < rows * columns; i++)
        {
            matrix[i] = rand() % 200 - 50; // [-50, 150)
        }
    }

    MPI_Bcast(matrix, rows * columns, MPI_INT, 0, MPI_COMM_WORLD);

    taskLength = columns / procNum;
    if (columns % procNum != 0)
    {
        taskLength++;
    }

    for (int i = taskLength * procRank; i < taskLength * (procRank + 1); i++)
    {
        min = matrix[i];
        for (int j = 0; j < rows; j++)
        {
            if (matrix[j * columns + i] < min)
            {
                min = matrix[j * columns + i];
            }
        }
        if (procRank == 0)
        {
            minsInColumns[i] = min;
        }
        else
        {
            MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (procRank == 0)
    {
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
        for (int i = 0; i < columns; i++)
        {
            minsInColumns[i] = matrix[i];
            for (int j = 0; j < rows; j++)
            {
                if (matrix[j * columns + i] < minsInColumns[i])
                {
                    minsInColumns[i] = matrix[j * columns + i];
                }
            }
        }
        endTime = MPI_Wtime();
        printf("Serial time: %f\n", endTime - startTime);
    }

    MPI_Finalize();
    //MPI END

    free(matrix);
    free(minsInColumns);

    return 0;
}
