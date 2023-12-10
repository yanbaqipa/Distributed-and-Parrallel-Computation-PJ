#include <cuda_runtime.h>

__global__ void matrixMulKernel(float *A, float *B, float *C, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    // 这个保护实际上不够精密，因为row实际上范围很小
    // 但是实在不想搞得太复杂
    // if (row < size && col < size)
    {
        for (int k = 0; k < size; k++)
        {
            sum += A[row * size + k] * B[k * size + col];
        }

        C[row * size + col] = sum;
    }
}

__global__ void matrixMulShared(float *A, float *B, float *C, int size)
{
    const int TILE_WIDTH = 16;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算全局索引
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // 遍历所有必要的矩阵A和B的块
    for (int m = 0; m < (size - 1) / TILE_WIDTH + 1; ++m)
    {
        // 加载A和B的子矩阵到共享内存
        // if (row < size && m * TILE_WIDTH + tx < size)
        As[ty][tx] = A[row * size + m * TILE_WIDTH + tx];
        // else
        //     As[ty][tx] = 0.0;

        // if (col < size && m * TILE_WIDTH + ty < size)
        Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * size + col];
        // else
        //     Bs[ty][tx] = 0.0;

        __syncthreads(); // 确保数据加载完成

        // 计算子矩阵乘积
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads(); // 确保所有线程完成计算
    }

    // 写入结果矩阵
    // if (row < size && col < size)
    C[row * size + col] = sum;
}

__global__ void matrixMulFour(float *A, float *B, float *C, int size)
{
    // const int BLOCK_WIDTH = 8;
    // const int TILE_WIDTH = 4;
    const int BLOCK_WIDTH = 16;
    const int TILE_WIDTH = 8;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算每个线程应该处理的子矩阵的起始行和列
    int rowStart = by * BLOCK_WIDTH + ty * 2;
    int colStart = bx * BLOCK_WIDTH + tx * 2;

    float Csub[2][2] = {0.0f}; // 用于存储计算结果的临时数组

    // 遍历所有必要的矩阵A和B的块
    for (int m = 0; m < (size - 1) / TILE_WIDTH + 1; ++m)
    {
        __shared__ float As[BLOCK_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][BLOCK_WIDTH];

        // 加载A和B的子矩阵到共享内存
        for (int i = 0; i < 2; ++i)
        {
            if (rowStart + i < size && m * TILE_WIDTH + tx < size)
                As[ty * 2 + i][tx] = A[(rowStart + i) * size + m * TILE_WIDTH + tx];
            else
                As[ty * 2 + i][tx] = 0.0;

            if (colStart + i < size && m * TILE_WIDTH + ty < size)
                Bs[ty][tx * 2 + i] = B[(m * TILE_WIDTH + ty) * size + colStart + i];
            else
                Bs[ty][tx * 2 + i] = 0.0;
        }

        __syncthreads();

        // 计算子矩阵乘积
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    Csub[i][j] += As[ty * 2 + i][k] * Bs[k][tx * 2 + j];
                }
            }
        }

        __syncthreads();
    }

    // 将计算结果写回全局内存
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            if (rowStart + i < size && colStart + j < size)
                C[(rowStart + i) * size + colStart + j] = Csub[i][j];
        }
    }
}

// __global__ void TmatrixMulKernel(float *A, float *B_transposed, float *C, int size)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0;
//     if (row < size && col < size)
//     { // 防止超出矩阵范围的安全检查
//         for (int k = 0; k < size; k++)
//         {
//             sum += A[row * size + k] * B_transposed[col * size + k];
//         }
//         C[row * size + col] = sum;
//     }
// }
// __global__ void TmatrixMulSharedKernel(float *A, float *B_transposed, float *C, int size)
// {
//     // 块大小，假设是16x16的块
//     const int blockSize = 16;
//     // 声明共享内存，用于存储块中的A和B的瓦片（Tile）
//     __shared__ float As[blockSize][blockSize];
//     __shared__ float Bs[blockSize][blockSize];
//     int blockX = blockIdx.x;
//     int blockY = blockIdx.y;
//     int threadX = threadIdx.x;
//     int threadY = threadIdx.y;
//     int row = blockY * blockSize + threadY;
//     int col = blockX * blockSize + threadX;
//     float sum = 0.0f;
//     // 遍历A和B的瓦片
//     for (int m = 0; m < (size / blockSize); ++m)
//     {
//         // 将瓦片加载到共享内存
//         As[threadY][threadX] = A[row * size + (m * blockSize + threadX)];
//         Bs[threadX][threadY] = B_transposed[col * size + (m * blockSize + threadY)];
//         // 同步以确保瓦片被完全加载
//         __syncthreads();
//         // 计算部分点积
//         for (int k = 0; k < blockSize; ++k)
//         {
//             sum += As[threadY][k] * Bs[threadX][k];
//         }
//         // 同步以确保所有线程完成计算前不会开始下一轮加载
//         __syncthreads();
//     }
//     // 写回结果
//     if (row < size && col < size)
//     {
//         C[row * size + col] = sum;
//     }
// }

extern "C" void runMatrixMultiplication(float *subA, float *B, float *subC, int rows_per_process, int N)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, rows_per_process * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, rows_per_process * N * sizeof(float));

    cudaMemcpy(d_A, subA, rows_per_process * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows_per_process + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // // matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // dim3 threadsPerBlock(4, 4);
    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid((N + 2 * threadsPerBlock.x - 1) / 2 * threadsPerBlock.x, (rows_per_process + 2 * threadsPerBlock.y - 1) / 2 * threadsPerBlock.y);
    matrixMulFour<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(subC, d_C, rows_per_process * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
