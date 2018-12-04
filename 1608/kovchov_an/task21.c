#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

void GenerateMatrix(double *matrix, int dimension);
void GenerateFreeMembers(double *vector, int dimension);

void LinearGaussMethod(double *coefficientMatrix, double *freeMembersColumn, int dimension, double *resultingValues);
int AreSolutionsEqual(double *linearSolutions, double *parallelSolutions, int dimension, double EPS);

void PrintLinearEquationSystem(double *coefficientMatrix, double *freeMembers, int dimension);
void PrintResultingValues(double *solutions, int dimension);

int main(int argc, char *argv[])
{
    double *coefficientMatrix;       // Матрица коэффициентов
    double *freeMembersColumn;       // Столбец свободных членов
    double *parallelResultingValues; // Значения неизвестных при паралелльных вычислениях
    double *linearResultingValues;   // Значения неизвестных при линейных вычислениях
    double *coefficientsPerProc;     // Подматрица коэффициентов, уникальная для каждого процесса
    double *freeMembersPerProc;      // Элементы столбца свободных членов, уникальные для каждого процесса
    double *resultingValuesPerProc;  // Элементы массива значений неизвестных на конкретном процессе, уникальные для каждого процесса
    double *pivotRow;                // Ведущая строка
    int *pivotRowIndexes;            // Номера ведущих строк системы на каждой итерации
    int *processIterationNumbers;    // Номера итераций, на которой строка выбрана в качестве ведущей
    int dimension;                   // Размерность исходной системы
    int rowsPerProc;                 // Размерность подматрицы коэффициентов, уникальная для каждого процесса
    int procRankWithPivotRow;        // Номер процесса, хранящего текущую ведущую строку
    int pivotRowIndexPerProc;        // Номер текущей ведущей строки на содержащем её процессе
    int currentPivotRowNumber;       // индекс ведущей строки на конкретном процессе
    int procNumber, procRank;
    int i, j, k;
    double startTime, finishTime;
    double multiplier, max;
    double currentResultingValue;

    struct
    {
        double value;
        int processRank;
    } localMax, globalMax;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNumber);

    if (procRank == 0)
    {
        dimension = (argc == 2) ? atoi(argv[1]) : 1000;

        coefficientMatrix = (double *)malloc(sizeof(double) * dimension * dimension);
        freeMembersColumn = (double *)malloc(sizeof(double) * dimension);

        srand(time(NULL));
        GenerateMatrix(coefficientMatrix, dimension);
        GenerateFreeMembers(freeMembersColumn, dimension);

        if (dimension < 10)
        {
            PrintLinearEquationSystem(coefficientMatrix, freeMembersColumn, dimension);
        }

        // Определение размера части данных, расположенных на конкретном процессе
        rowsPerProc = dimension / procNumber;
    }

    startTime = MPI_Wtime();

    MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    coefficientsPerProc = (double *)malloc(sizeof(double) * rowsPerProc * dimension);
    freeMembersPerProc = (double *)malloc(sizeof(double) * rowsPerProc);

    // Рассылка матрицы процессам
    MPI_Scatter(coefficientMatrix, rowsPerProc * dimension, MPI_DOUBLE, coefficientsPerProc, rowsPerProc * dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Рассылка столбца свободных членов
    MPI_Scatter(freeMembersColumn, rowsPerProc, MPI_DOUBLE, freeMembersPerProc, rowsPerProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    pivotRow = (double *)malloc(sizeof(double) * (dimension) + 1); // +1 для свободного члена
    pivotRowIndexes = (int *)malloc(sizeof(int) * dimension);
    processIterationNumbers = (int *)malloc(sizeof(int) * rowsPerProc);

    if (procRank == 0)
    {
        for (i = 0; i < rowsPerProc; i++)
        {
            processIterationNumbers[i] = -1;
        }
    }

    MPI_Bcast(processIterationNumbers, rowsPerProc, MPI_INT, 0, MPI_COMM_WORLD);

    // Цикл гауссовых преобразований
    for (i = 0; i < dimension; i++)
    {
        // Определение локальной ведущей строки
        max = 0;
        for (j = 0; j < rowsPerProc; j++)
        {
            if ((processIterationNumbers[j] == -1) &&
                (max < abs(coefficientsPerProc[i + dimension * j])))
            {
                max = abs(coefficientsPerProc[i + dimension * j]);
                currentPivotRowNumber = j;
            }
        }

        // Запоминание локальных максимальных значений и рангов соответствующих процессов в единую структуру
        localMax.value = max;
        localMax.processRank = procRank;
        // Определение максимального по модулю элемента
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Вычисление номера ведущей строки всей системы
        if (procRank == globalMax.processRank)
        {
            if (globalMax.value == 0)
            {
                processIterationNumbers[rowsPerProc] = i;
            }
            else
            {
                processIterationNumbers[currentPivotRowNumber] = i;
            }
            pivotRowIndexes[i] = rowsPerProc * procRank + currentPivotRowNumber;
        }

        // Рассылка всем процессам вычисленного номера ведущей строки системы
        MPI_Bcast(&pivotRowIndexes[i], 1, MPI_INT, globalMax.processRank, MPI_COMM_WORLD);
        if (procRank == globalMax.processRank)
        {
            // Заполнение ведущей строки системы
            for (j = 0; j < dimension; j++)
            {
                pivotRow[j] = coefficientsPerProc[currentPivotRowNumber * dimension + j];
            }
            pivotRow[dimension] = freeMembersPerProc[currentPivotRowNumber];
        }

        // Рассылка ведущей строки всем процессам
        MPI_Bcast(pivotRow, dimension + 1, MPI_DOUBLE, globalMax.processRank, MPI_COMM_WORLD);

        // Исключение неизвестных в столбце с номером i
        for (j = 0; j < rowsPerProc; j++)
        {
            if (processIterationNumbers[j] == -1)
            {
                multiplier = coefficientsPerProc[j * dimension + i] / pivotRow[i];
                for (k = i; k < dimension; k++)
                {
                    coefficientsPerProc[j * dimension + k] -= pivotRow[k] * multiplier;
                }
                freeMembersPerProc[j] -= pivotRow[dimension] * multiplier;
            }
        }
    }

    resultingValuesPerProc = (double *)malloc(sizeof(double) * rowsPerProc);

    // Основной цикл обратного вычислительного процесса
    for (i = dimension - 1; i >= 0; i--)
    {
        currentResultingValue = 0.0;

        // Определение ранга процесса, содержащего текущую ведущую строку, и номера этой строки на процессе
        for (j = 0; j < procNumber - 1; j++)
        {
            if ((pivotRowIndexes[i] >= rowsPerProc * j) && (pivotRowIndexes[i] < rowsPerProc * (j + 1)))
            {
                procRankWithPivotRow = j;
            }
        }
        if (pivotRowIndexes[i] >= rowsPerProc * (procNumber - 1))
        {
            procRankWithPivotRow = procNumber - 1;
        }

        // Определение номера строки на процессе
        pivotRowIndexPerProc = pivotRowIndexes[i] - rowsPerProc * procRankWithPivotRow;

        // Вычисление значения неизвестной
        if (procRank == procRankWithPivotRow)
        {
            if (coefficientsPerProc[pivotRowIndexPerProc * dimension + i] == 0)
            {
                if (freeMembersPerProc[pivotRowIndexPerProc] == 0)
                {
                    currentResultingValue = 0.0;
                }
            }
            else
            {
                currentResultingValue = freeMembersPerProc[pivotRowIndexPerProc] / coefficientsPerProc[pivotRowIndexPerProc * dimension + i];
            }
            resultingValuesPerProc[pivotRowIndexPerProc] = currentResultingValue;
        }

        // Рассылка всем остальным процессам найденного значения переменной
        MPI_Bcast(&currentResultingValue, 1, MPI_DOUBLE, procRankWithPivotRow, MPI_COMM_WORLD);

        // Корректировка элементов вектора свободных членов
        for (j = 0; j < rowsPerProc; j++)
        {
            if (processIterationNumbers[j] < i)
            {
                freeMembersPerProc[j] -= coefficientsPerProc[dimension * j + i] * currentResultingValue;
            }
        }
    }

    free(coefficientsPerProc);
    free(freeMembersPerProc);
    free(pivotRowIndexes);
    free(processIterationNumbers);

    parallelResultingValues = (double *)malloc(sizeof(double) * dimension);
    MPI_Gather(resultingValuesPerProc, rowsPerProc, MPI_DOUBLE, parallelResultingValues, rowsPerProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    free(resultingValuesPerProc);
    finishTime = MPI_Wtime();

    if (procRank == 0)
    {
        if (dimension < 10)
        {
            printf("\nParallel solutions: ");
            PrintResultingValues(parallelResultingValues, dimension);
        }
        printf("Parallel time: %.4lf\n", finishTime - startTime);

        linearResultingValues = (double *)malloc(sizeof(double) * dimension);
        startTime = MPI_Wtime();
        LinearGaussMethod(coefficientMatrix, freeMembersColumn, dimension, linearResultingValues);
        finishTime = MPI_Wtime();

        free(coefficientMatrix);
        free(freeMembersColumn);

        if (dimension < 10)
        {
            printf("\nLinear solutions: ");
            PrintResultingValues(linearResultingValues, dimension);
        }
        printf("Linear time: %.4lf\n", finishTime - startTime);

        if (!AreSolutionsEqual(linearResultingValues, parallelResultingValues, dimension, 0.00000001))
        {
            printf("Error in calculations! Linear and parallel solutions are not equal\n");
        }

        free(linearResultingValues);
        free(parallelResultingValues);
    }

    MPI_Finalize();

    return 0;
}

void GenerateMatrix(double *matrix, int dimension)
{
    int i;

    for (i = 0; i < dimension * dimension; i++)
    {
        matrix[i] = rand() % 30 + 1;
    }
}

void GenerateFreeMembers(double *vector, int dimension)
{
    int i;

    for (i = 0; i < dimension; i++)
    {
        vector[i] = rand() % 30 + 1;
    }
}

void LinearGaussMethod(double *coefficientMatrix, double *freeMembersColumn, int dimension, double *resultingValues)
{
    int i, j, k;
    int currentPivotRowNumber = 0; // Номер ведущей строки в матрице коэффициентов
    double pivotValue;             // Разрешающий элемент, i-тый элемент ведущей строки
    double max, multiplier;
    int *pivotRowNumbers = (int *)malloc(sizeof(int) * dimension);
    int *iterationNumbers = (int *)malloc(sizeof(int) * dimension);

    for (i = 0; i < dimension; i++)
    {
        pivotRowNumbers[i] = 0;
        iterationNumbers[i] = -1;
    }

    // Прямой ход Гаусса
    for (i = 0; i < dimension; i++)
    {
        // Определение ведущей строки
        max = 0.0;

        for (j = 0; j < dimension; j++)
        {
            // Определение максимального по модулю элемента в j-том столбце
            if ((iterationNumbers[j] == -1) && (abs(coefficientMatrix[j * dimension + i]) > max))
            {
                currentPivotRowNumber = j;
                max = coefficientMatrix[j * dimension + i];
            }
        }

        // Вычисление номера ведущей строки и разрешающего элемента
        pivotRowNumbers[i] = currentPivotRowNumber;
        iterationNumbers[currentPivotRowNumber] = i;
        pivotValue = coefficientMatrix[currentPivotRowNumber * dimension + i];

        //Исключение неизвестных в j-том столбце
        for (j = 0; j < dimension; j++)
        {
            if (iterationNumbers[j] == -1)
            {
                multiplier = coefficientMatrix[j * dimension + i] / pivotValue;
                for (k = i; k < dimension; k++)
                {
                    coefficientMatrix[j * dimension + k] -= multiplier * coefficientMatrix[currentPivotRowNumber * dimension + k];
                }
                freeMembersColumn[j] -= multiplier * freeMembersColumn[currentPivotRowNumber];
            }
        }
    }
    free(iterationNumbers);

    // Обратный ход Гаусса
    for (i = dimension - 1; i >= 0; i--)
    {
        // Вычисление значение неизвестных
        resultingValues[i] = freeMembersColumn[pivotRowNumbers[i]] / coefficientMatrix[dimension * pivotRowNumbers[i] + i];
        coefficientMatrix[pivotRowNumbers[i] * dimension + i] = 1;

        //Корректировка элементов вектора свободных членов
        for (j = 0; j < i; j++)
        {
            freeMembersColumn[pivotRowNumbers[j]] -= coefficientMatrix[pivotRowNumbers[j] * dimension + i] * resultingValues[i];
            // Зануление всех элементов матрицы выше главной диагонали
            coefficientMatrix[pivotRowNumbers[j] * dimension + i] = 0;
        }
    }

    free(pivotRowNumbers);
}

int AreSolutionsEqual(double *linearSolutions, double *parallelSolutions, int dimension, double EPS)
{
    int i, j, matchCounter = 0;

    for (i = 0; i < dimension; i++)
    {
        for (j = 0; j < dimension; j++)
        {
            if (fabs(parallelSolutions[i] - linearSolutions[j]) <= EPS)
            {
                matchCounter++;
            }
        }
    }

    return (matchCounter == dimension) ? 1 : 0;
}

void PrintLinearEquationSystem(double *coefficientMatrix, double *freeMembers, int dimension)
{
    int i, j;

    printf("Linear equation system\n");
    for (i = 0; i < dimension; i++)
    {
        for (j = 0; j < dimension; j++)
        {
            printf("%4.0lfx%d", coefficientMatrix[dimension * i + j], j + 1);
            (j != dimension - 1) ? printf("   + ") : printf(" ");
        }
        printf("= %.0lf\n", freeMembers[i]);
    }
}

void PrintResultingValues(double *resultingValues, int dimension)
{
    for (int i = 0; i < dimension; i++)
    {
        printf("%.4lf ", resultingValues[i]);
    }
    printf("\n");
}
