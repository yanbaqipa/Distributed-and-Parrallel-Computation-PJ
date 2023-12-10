#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <cstring>

extern "C" void runMatrixMultiplication(float *subA, float *B, float *subC, int rows_per_process, int N);

const int N = 4095;

void initMatrix(float *M, int size)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < size * size; ++i)
    {
        M[i] = dis(gen);
    }
}

void printMatrix(float *M, int size)
{
    for (int i = 0; i < size * size; i++)
    {
        std::cout << M[i] << " ";
    }
    std::cout << "\n";
}

void transposeMatrix(float *B, float *T, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            T[j * N + i] = B[i * N + j]; // T的行列索引与B相反
        }
    }
}

void padMatrix(float *&M, int dim, int targetDim)
{
    float *newM = new float[targetDim * targetDim];

    std::fill_n(newM, targetDim * targetDim, 0.0f);

    for (int r = 0; r < dim; ++r)
    {
        std::memcpy(newM + r * targetDim, M + r * dim, dim * sizeof(float));
    }

    delete[] M;

    M = newM;
}

void trimPadding(float *&C, int N, int _N)
{
    float *oriC = new float[N * N];

    for (int i = 0; i < N; ++i)
    {
        std::memcpy(oriC + i * N, C + i * _N, N * sizeof(float));
    }

    delete[] C;

    C = oriC;
}

void matrixMultiplyCPU(float *A, float *B, float *C, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k)
            {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

bool compareMatrices(float *matrix1, float *matrix2, int size)
{
    float epsilon = 1e-5; // 设置一个合适的epsilon值
    for (int i = 0; i < size * size; ++i)
    {
        if (fabs(matrix1[i] - matrix2[i]) > epsilon)
        {
            return false; // 矩阵不相等
        }
    }
    return true; // 矩阵相等
}

int main(int argc, char **argv)
{
    // int gpu_id = 1; // 默认选择第一个GPU
    // if (argc > 1)
    // {
    //     gpu_id = atoi(argv[1]);
    // }
    // cudaSetDevice(rank);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank + 2);

    float *A, *B, *C;
    float *subA, *subC;
    float *STD;

    // Root process initializes data
    if (rank == 0)
    {
        A = new float[N * N];
        B = new float[N * N];

        initMatrix(A, N);
        initMatrix(B, N);

        // STD = new float[N * N];
        // matrixMultiplyCPU(A, B, STD, N);
    }
    // 生成完例子，开始计时
    double start = MPI_Wtime();

    // 填充矩阵方便计算
    int _N = ((N + size * 16 - 1) / (size * 16)) * (size * 16);
    if (rank == 0)
    {
        std::cout << "Padded size: " << _N << std::endl;
        padMatrix(A, N, _N);
        padMatrix(B, N, _N);
        C = new float[_N * _N];
    }

    if (rank != 0)
    {
        B = new float[_N * _N];
    }

    int *sendcounts = new int[size]; // 每个进程应接收的元素数量
    int *displs = new int[size];     // 每个进程接收的元素在发送缓冲区中的偏移

    int rowsPerProcess = _N / size; // _N是填充后的行数，应该能被进程数整除

    for (int i = 0; i < size; ++i)
    {
        sendcounts[i] = rowsPerProcess * _N;
        displs[i] = i * sendcounts[i];
    }

    // 使用MPI_Scatterv向各个子进程发送数据
    subA = new float[sendcounts[rank]];
    subC = new float[sendcounts[rank]];
    MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, subA, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // int baseRowsPerProcess = _N / size;
    // int extraRows = _N % size;
    // int offset = 0;
    // for (int i = 0; i < size; ++i)
    // {
    //     int rowsToSend = (i < extraRows) ? (baseRowsPerProcess + 1) : baseRowsPerProcess;
    //     sendcounts[i] = rowsToSend * N; // N是A的列数
    //     displs[i] = offset;
    //     offset += sendcounts[i];
    // }
    // // 使用MPI_Scatterv向各个子进程发送数据
    // subA = new float[sendcounts[rank]];
    // subC = new float[sendcounts[rank]];
    // MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, subA, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // broadcast matrix B
    // MPI_Scatter(A, rows_per_process * N, MPI_FLOAT, subA, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, _N * _N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 调用CUDA矩阵乘法
    if (rank == 0)
    {
        std::cout << "row per process: " << sendcounts[rank] / _N << std::endl;
    }
    double cudastart = MPI_Wtime();
    // runMatrixMultiplication(subA, B, subC, sendcounts[rank] / _N, _N);
    runMatrixMultiplication(subA, B, subC, sendcounts[rank] / _N, _N);
    double cudaduration = MPI_Wtime() - cudastart;
    if (rank == 0)
    {
        std::cout << "CUDA Time cost: " << cudaduration << std::endl;
    }

    // 主进程收集数据并去除填充
    if (rank == 0)
    {
        MPI_Gatherv(subC, sendcounts[rank], MPI_FLOAT, C, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
        trimPadding(C, N, _N);
    }
    else
    {
        MPI_Gatherv(subC, sendcounts[rank], MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    // 处理完毕，结束计时
    double duration = MPI_Wtime() - start;

    delete[] sendcounts;
    delete[] displs;

    if (rank == 0)
    {
        std::cout << "Time cost:" << duration << std::endl;

        // if (!compareMatrices(STD, C, N))
        // {
        //     std::cout << "Failed!" << std::endl;
        //     printMatrix(C, N);
        //     printMatrix(STD, N);
        // }
        // else
        // {
        //     std::cout << "Successful!" << std::endl;
        // }
        // delete[] STD;
        delete[] A;
        delete[] C;
    }

    delete[] subA;
    delete[] subC;
    delete[] B;

    MPI_Finalize();

    return 0;
}
