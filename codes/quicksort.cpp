#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <fstream>
#define Finalthreshold 2000

// 自然分布的测试样例
std::vector<int> NaturalSample(int length, int seed)
{
    std::vector<int> result(length);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<> distribution(0, 1000);

    for (int i = 0; i < length; ++i)
    {
        result[i] = distribution(generator);
    }

    return result;
}

// 比较有序的测试样例
std::vector<int> OrderedSample(int length, int seed)
{
    std::vector<int> result(length);

    for (int i = 0; i < length; ++i)
    {
        result[i] = i;
    }

    std::mt19937 generator(seed);
    std::uniform_int_distribution<> distribution(0, length - 1);
    int num_swaps = length * 0.05;

    for (int i = 0; i < num_swaps; ++i)
    {
        int pos1 = distribution(generator);
        int pos2 = distribution(generator);
        std::swap(result[pos1], result[pos2]);
    }

    return result;
}

// 三数取中
int medianOfThree(int *array, int low, int high)
{
    int mid = low + (high - low) / 2;
    if (array[mid] < array[low])
    {
        std::swap(array[low], array[mid]);
    }
    if (array[high] < array[low])
    {
        std::swap(array[low], array[high]);
    }
    if (array[mid] < array[high])
    {
        std::swap(array[mid], array[high]);
    }
    return array[high];
}

// 快排核心
int partition(int *array, int low, int high)
{
    int pivot = medianOfThree(array, low, high);
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)
    {
        if (array[j] < pivot)
        {
            i++;
            std::swap(array[i], array[j]);
        }
    }
    std::swap(array[i + 1], array[high]);
    return (i + 1);
}

// 插入排序
void Insertionsort(int *array, int low, int high)
{
    for (int i = low + 1; i <= high; i++)
    {
        int key = array[i];
        int j = i - 1;
        while (j >= low && array[j] > key)
        {
            array[j + 1] = array[j];
            j--;
        }
        array[j + 1] = key;
    }
}

// 非并行化快排
void leafQuicksort(int *array, int low, int high)
{
    // 末端采用插入排序
    if (high - low < Finalthreshold)
    {
        Insertionsort(array, low, high);
    }
    else if (low < high)
    {
        int pi = partition(array, low, high);
        leafQuicksort(array, low, pi - 1);
        leafQuicksort(array, pi + 1, high);
    }
}

// 并行化快排
void parallelQuicksort(int *array, int low, int high, int depth)
{
    // 末端采用插入排序
    if (high - low < Finalthreshold)
    {
        Insertionsort(array, low, high);
    }
    else if (low < high)
    {
        int pi = partition(array, low, high);
        int len = high - low + 1;
        // 在递归太深时，选择非并行化快排
        if (depth <= 0)
        {
            leafQuicksort(array, low, pi - 1);
            leafQuicksort(array, pi + 1, high);
        }
        else
#pragma omp parallel sections
        {
#pragma omp section
            {
                parallelQuicksort(array, low, pi - 1, depth - 1);
            }
#pragma omp section
            {
                parallelQuicksort(array, pi + 1, high, depth - 1);
            }
        }
    }
}

// 检查是否有序
bool checkSorted(int *array, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        if (array[i] > array[i + 1])
        {
            return false;
        }
    }
    return true;
}

// 直接比较vector以进行完备的检查
bool cmpvec(const std::vector<int> &data, const std::vector<int> &std)
{
    if (data.size() != std.size())
        return false;
    for (int i = 0; i < data.size(); i++)
    {
        if (data[i] != std[i])
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Not enough arguments!";
        exit(0);
    }
    int len = std::stoi(argv[1]);
    bool Ordered = (argv[2][0] == 'O');
    int max_depth = std::stoi(argv[3]);
    // std::cout << argv[2] << std::endl;
    // std::cout << Ordered << std::endl;

    // 生成测试数据
    std::vector<int>
        sample = NaturalSample(len, 0);
    if (Ordered)
    {
        sample = OrderedSample(len, 0);
    }
    int *p = sample.data();
    int n = sample.size();
    std::vector<int> std_sorted = sample;
    std::sort(std_sorted.begin(), std_sorted.end());

    // 运行快排并计时
    auto start = std::chrono::high_resolution_clock::now();
    parallelQuicksort(p, 0, n - 1, max_depth);
    auto end = std::chrono::high_resolution_clock::now();

    // 输出结果
    if (!cmpvec(sample, std_sorted))
    {
        std::cout << "Sorting not successful" << std::endl;
        exit(0);
    }
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time cost:" << elapsed.count() << std::endl;
    // std::cout << "Len data:" << sample.size() << std::endl;
    // std::ofstream logFile("r4_sizevar_log.txt", std::ios_base::app);
    // if (!logFile.is_open())
    // {
    //     std::cerr << "Error: Unable to open log.txt for writing." << std::endl;
    //     return 1;
    // }
    // logFile << "Ordered " << sample.size() << " " << elapsed.count() << std::endl;
    return 0;
}
