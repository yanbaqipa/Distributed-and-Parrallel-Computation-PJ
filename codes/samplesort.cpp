#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <random>
#include <chrono>
#include <string>
#include <fstream>

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

// 检查是否有序
bool checkSorted(const std::vector<int> &data)
{
    for (int i = 0; i < data.size() - 1; i++)
    {
        if (data[i] > data[i + 1])
        {
            return false;
        }
    }
    return true;
}

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
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> data;
    // std::vector<int> std_sorted;
    int datasize;
    if (rank == 0)
    {
        int len = std::stoi(argv[1]);
        bool Ordered = (argv[2][0] == 'O');
        data = NaturalSample(len, 0);
        if (Ordered)
        {
            data = OrderedSample(len, 0);
        }
        datasize = data.size();
        // std_sorted = data;
        // std::sort(std_sorted.begin(), std_sorted.end());
    }
    // 生成完例子，开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 添加若干个0，保证能均匀分配
    // 在实际应用中，可以添加若干个对应类型的对象，只需要有正确的==运算符用以后续移除即可
    int padding = 0;
    if (rank == 0)
    {
        padding = size - datasize % size;
        if (padding == size)
        {
            padding = 0;
        }
        if (padding != 0)
        {
            for (int i = 0; i < padding; i++)
            {
                data.push_back(0);
            }
            datasize += padding;
        }
    }

    MPI_Bcast(&datasize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int n_per_proc = datasize / size;
    std::vector<int> local_data(n_per_proc);

    // DONE:测试点：完成分配
    MPI_Scatter(&data[0], n_per_proc, MPI_INT, &local_data[0], n_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    // std::cout << "rank" << rank << ": ";
    // for (int i = 0; i < local_data.size(); i++)
    // {
    //     std::cout << local_data[i] << " ";
    // }
    // std::cout << std::endl;

    // 各个进程本地排序
    std::sort(local_data.begin(), local_data.end());

    // 从各个进程中采样 regular samples
    int step = n_per_proc / size;
    std::vector<int> samples;
    for (int i = 0; i < n_per_proc; i += step)
    {
        samples.push_back(local_data[i]);
    }

    std::vector<int> all_samples(size * size);
    MPI_Allgather(&samples[0], size, MPI_INT, &all_samples[0], size, MPI_INT, MPI_COMM_WORLD);

    // 主进程排序所有样本并选出分界点
    if (rank == 0)
    {
        std::sort(all_samples.begin(), all_samples.end());
    }

    std::vector<int> pivots(size - 1);
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            pivots[i - 1] = all_samples[i * size];
        }
    }

    // 广播pivots到所有进程
    MPI_Bcast(&pivots[0], pivots.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // DONE:测试点：收集和分析样本
    // std::cout << "rank" << rank << ": ";
    // for (int i = 0; i < pivots.size(); i++)
    // {
    //     std::cout << pivots[i] << " ";
    // }
    // std::cout << std::endl;

    // 根据分界点将数据划分到各个bucket
    std::vector<std::vector<int>> buckets(size);
    int pivot_idx = 0;
    for (int num : local_data)
    {
        while (pivot_idx < pivots.size() && num > pivots[pivot_idx])
        {
            pivot_idx++;
        }
        buckets[pivot_idx].push_back(num);
    }

    // 各进程发送bucket大小
    std::vector<int> bucket_sizes(size);
    for (int i = 0; i < size; i++)
    {
        bucket_sizes[i] = buckets[i].size();
    }

    // std::cout << "rank" << rank << ": ";
    // for (int i = 0; i < bucket_sizes.size(); i++)
    // {
    //     std::cout << bucket_sizes[i] << " ";
    // }
    // std::cout << std::endl;

    std::vector<int> all_bucket_sizes(size * size);
    MPI_Allgather(&bucket_sizes[0], size, MPI_INT, &all_bucket_sizes[0], size, MPI_INT, MPI_COMM_WORLD);

    // DONE:测试点：收集bucket size
    // if (rank == 0)
    // {
    //     std::cout << "ALL sizes:" << std::endl;
    //     for (int i = 0; i < size; i++)
    //     {
    //         for (int j = 0; j < size; j++)
    //         {
    //             int id = i * size + j;
    //             std::cout << all_bucket_sizes[id] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // std::cout << "Begin Exchange!" << std::endl;
    // 各进程发送数据给对应的进程
    std::vector<int> recv_sizes(size);
    for (int i = 0; i < size; i++)
    {
        recv_sizes[i] = all_bucket_sizes[i * size + rank];
    }
    std::vector<int> sdispls(size), rdispls(size);
    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 1; i < size; i++)
    {
        sdispls[i] = sdispls[i - 1] + bucket_sizes[i - 1];
        rdispls[i] = rdispls[i - 1] + recv_sizes[i - 1];
    }
    int total_recv = rdispls[size - 1] + recv_sizes[size - 1];
    std::vector<int> recv_data(total_recv);
    MPI_Alltoallv(&local_data[0], &bucket_sizes[0], &sdispls[0], MPI_INT, &recv_data[0], &recv_sizes[0], &rdispls[0], MPI_INT, MPI_COMM_WORLD);
    // std::cout << "Done Exchange!" << std::endl;

    // 测试点：完成交换
    // std::cout << "rank" << rank << ": ";
    // for (int i = 0; i < recv_data.size(); i++)
    // {
    //     std::cout << recv_data[i] << " ";
    // }
    // std::cout << std::endl;

    // 各进程继续排序数据
    std::stable_sort(recv_data.begin(), recv_data.end());

    // std::cout << "Before Gathering!" << std::endl;
    int final_count = recv_data.size();
    std::vector<int> final_counts(size);
    MPI_Gather(&final_count, 1, MPI_INT, final_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> final_displs(size);
    if (rank == 0)
    {
        final_displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            final_displs[i] = final_displs[i - 1] + final_counts[i - 1];
        }
    }

    // 使用MPI_Gatherv汇集所有进程的recv_data到主进程
    // std::cout << "Begin Gathering!" << std::endl;
    MPI_Gatherv(recv_data.data(), final_count, MPI_INT,
                data.data(), final_counts.data(), final_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    // std::cout << "Done Gathering!" << std::endl;
    // 收集完毕，结束计时
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        auto padding_start = std::find(data.begin(), data.end(), 0);
        auto padding_end = padding_start + padding;
        data.erase(padding_start, padding_end);
        // if (!cmpvec(data, std_sorted))
        // {
        //     std::cout << "Sorting not successful" << std::endl;
        //     exit(0);
        // }
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time cost:" << elapsed.count() << std::endl;
        // std::cout << "Len data:" << data.size() << std::endl;
        // std::ofstream logFile("log.txt", std::ios_base::app);
        // if (!logFile.is_open())
        // {
        //     std::cerr << "Error: Unable to open log.txt for writing." << std::endl;
        //     return 1;
        // }
        // logFile << "Ordered " << size << " " << elapsed.count() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
