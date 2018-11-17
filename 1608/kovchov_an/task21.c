#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Вариант 21: Метод Гаусса – ленточная вертикальная схема

void LinearGaussMethod(int *matrix, int size, int* resultMatrix)
{
    int i, j;
    int iterator = 0;
    int pivotRowNumber = 0; //TODO: whats this
    int *pivotPositions = NULL, *pivotIterators = NULL;

    pivotPositions = (int *)malloc(sizeof(int) * size);
    pivotIterators = (int *)malloc(sizeof(int) * size);

    for (i = 0; i < size; i++)
    {
        pivotPositions[i] = 0;  //TODO: try pivotPositions = {0}
        pivotIterators[i] = -1;
    }

    for (iterator = 0; iterator < size; iterator++)
    {
        double maxValue = 0;

        for (i = 0; i < size; i++)
        {
            if ((pivotIterators[i] == -1) && (fabs(matrix[i * (size + 1) + iterator]) > maxValue))
            {
                pivotRowNumber = i;
                maxValue = matrix[i * (size + 1) + iterator];
            }
        }

        pivotPositions[iterator] = pivotRowNumber;
        pivotIterators[pivotRowNumber] = iterator;

        double pivotFactor; //TODO: whats this
        double pivotValue = matrix[pivotRowNumber * (size + 1) + iterator];

        for (i = 0; i < size; i++)
        {
            if (pivotIterators[i] == -1)
            {
                pivotFactor = matrix[i * (size + 1) + iterator] / pivotValue;
                for (j = iterator; j < size; j++)
                {
                    matrix[i * (size + 1) + j] -= pivotFactor * matrix[pivotRowNumber * (size + 1) + j];
                }
                matrix[i * (size + 1) + size] -= pivotFactor * matrix[pivotRowNumber * (size + 1) + size];
            }
        }
    }

    int rowIndex, row;

    for (i = size - 1; i >= 0; i--)
    {
        rowIndex = pivotPositions[i];
        resultMatrix[i] = matrix[rowIndex * (size + 1) + size] / matrix[(size + 1) * rowIndex + i];
        matrix[rowIndex * (size + 1) + i] = 1;

        for (j = 0; j < i; j++)
        {
            row = pivotPositions[j];
            matrix[row * (size + 1) + size] -= matrix[row * (size + 1) + i] * resultMatrix[i];
            matrix[row * (size + 1) + i] = 0;
        }
    }

    for (i = 0; i < size; i++)
    {
        printf("%f ", round(resultMatrix[i] * 1000) / 1000.0);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    int matrixSize;
    int procRank = 0, procNum = 1;
    int *matrix = NULL;
    int *linearResultMatrix = NULL, *parallelResultMatrix = NULL;
    int i;
    double startTime = 0.0, endTime = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (procRank == 0)
    {
        if (argc == 2)
        {
            matrixSize = atoi(argv[1]);
        }
        else
        {
            matrixSize = 10;
        }

        matrix = (int *)malloc(sizeof(int) * matrixSize * (matrixSize + 1));
        linearResultMatrix = (int *)malloc(sizeof(int) * matrixSize);

        srand(time(NULL));
        for (i = 0; i < matrixSize * (matrixSize + 1); i++)
        {
            matrix[i] = rand() % 100 + 50;
        }

        LinearGaussMethod(matrix, matrixSize, linearResultMatrix);

        free(matrix);
        free(linearResultMatrix);
    }

    return 0;
}