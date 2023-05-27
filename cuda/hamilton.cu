#ifdef __NVCC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include <chrono>

#define swap(a, b) \
    {              \
        a = a + b; \
        b = a - b; \
        a = a - b; \
    }

using namespace std;

bool check_hamiltonian_cycle_s(bool **graph, int *path, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        if (graph[path[i]][path[i + 1]] == false)
        {
            return false;
        }
    }
    if (graph[path[n - 1]][path[0]] == false)
    {
        return false;
    }
    return true;
}

bool check_partial_hamiltonian_cycle_s(bool **graph, int *path, int position)
{
    for (int i = 0; i < position - 1; i++)
    {
        if (graph[path[i]][path[i + 1]] == false)
        {
            return false;
        }
    }
    return true;
}

int find_hamiltonian_cycles_s(bool **graph, int *path, int position, int n)
{
    int cycles = 0;
    if (position == n - 2)
    {
        cycles += check_hamiltonian_cycle_s(graph, path, n);
        swap(path[n - 2], path[n - 1]);
        cycles += check_hamiltonian_cycle_s(graph, path, n);
        swap(path[n - 2], path[n - 1]);
        return cycles;
    }
    // if (!check_partial_hamiltonian_cycle_s(graph, path, position))
    // {
    //     return cycles;
    // }
    for (int i = position; i < n; i++)
    {
        if (i != position)
        {
            swap(path[position], path[i]);
        }
        cycles += find_hamiltonian_cycles_s(graph, path, position + 1, n);
        if (i != position)
        {
            swap(path[position], path[i]);
        }
    }
    return cycles;
}

int find_hamiltonian_cycles_s(bool **graph, int n)
{
    int *path = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        path[i] = i;
    }
    int result = find_hamiltonian_cycles_s(graph, path, 1, n);
    free(path);
    return result;
}

#ifdef __NVCC__
// I had to give up vectors and use arrays because of CUDA
__device__ bool check_hamiltonian_cycle_r(bool *__restrict__ *__restrict__ graph, int *path, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        if (graph[path[i]][path[i + 1]] == false)
        {
            return false;
        }
    }
    if (graph[path[n - 1]][path[0]] == false)
    {
        return false;
    }
    return true;
}

__device__ bool check_partial_hamiltonian_cycle_r(bool *__restrict__ *__restrict__ graph, int *path, int position)
{
    for (int i = 0; i < position - 1; i++)
    {
        if (graph[path[i]][path[i + 1]] == false)
        {
            return false;
        }
    }
    return true;
}

__device__ int find_hamiltonian_cycles_r(bool *__restrict__ *__restrict__ graph, int *path, int position, int level, int *acc, const int n, const int num_of_threads)
{
    int thread_id = threadIdx.x;
    int result = 0;
    if (position == n - 2)
    {
        result += check_hamiltonian_cycle_r(graph, path, n);
        swap(path[n - 2], path[n - 1]);
        result += check_hamiltonian_cycle_r(graph, path, n);
        swap(path[n - 2], path[n - 1]);
        return result;
    }
    // if (!check_partial_hamiltonian_cycle_r(graph, path, position))
    // {
    //     return result;
    // }
    for (int i = position; i < n; i++)
    {
        if (i != position)
        {
            swap(path[position], path[i]);
        }
        if (position == level)
        {
            (*acc)++;
            if ((*acc) % num_of_threads == thread_id)
            {
                result += find_hamiltonian_cycles_r(graph, path, position + 1, level, acc, n, num_of_threads);
            }
        }
        else
        {
            result += find_hamiltonian_cycles_r(graph, path, position + 1, level, acc, n, num_of_threads);
        }
        if (i != position)
        {
            swap(path[position], path[i]);
        }
    }
    return result;
}

__global__ void find_hamiltonian_cycles_r(bool *__restrict__ *__restrict__ graph, int **paths, int level, const int n, int *results, const int num_of_threads)
{
    int thread_id = threadIdx.x;
    int *path = paths[thread_id];
    int acc = 0;
    int thread_result = find_hamiltonian_cycles_r(graph, path, 1, level, &acc, n, num_of_threads);
    results[thread_id] = thread_result;
}
#endif

void generate_graph(bool **graph, int n, double probability)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double random = (double)rand() / (double)RAND_MAX;
            if (random < probability)
            {
                graph[i][j] = 1;
            }
            else
            {
                graph[i][j] = 0;
            }
        }
    }
}

void free_graph(bool **graph, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(graph[i]);
    }
    free(graph);
}

int main(int argc, char const *argv[])
{
    string type = "c";
    int n = 10;
    int level = 1;
    int number_of_threads = 10;
    if (argc == 1)
    {
        cout << "Usage: " << argv[0] << " [type] [n] ([probability]|[file]) [level] [number_of_threads]" << endl
             << endl;
        cout << "type:" << endl;
        cout << "c - compare results (random), s - serial (random), p - parallel (random)" << endl;
        cout << "fc - compare results (with file), fs - serial (with file), fp - parallel (with file)" << endl
             << endl;
        cout << "n: number of vertices" << endl
             << endl;
        cout << "for types c, s, p:" << endl;
        cout << "   probability: probability of edge between two vertices" << endl
             << endl;
        cout << "for types fc, fs, fp:" << endl;
        cout << "   file: path to file with graph" << endl
             << endl;
        cout << "level: level of recursion" << endl;
        cout << "number_of_threads: number of threads" << endl;
        return EXIT_FAILURE;
    }
    if (argc > 1)
    {
        type = argv[1];
    }
    if (type != "c" && type != "s" && type != "p" && type != "fc" && type != "fs" && type != "fp")
    {
        cout << "Type must be c, s, p, fc, fs or fp" << endl;
        return EXIT_FAILURE;
    }
    if (argc > 2)
    {
        n = atoi(argv[2]);
    }
    bool **graph = (bool **)malloc(n * sizeof(bool *));
    for (int i = 0; i < n; i++)
    {
        graph[i] = (bool *)malloc(n * sizeof(bool));
    }
    if (type == "c" || type == "s" || type == "p")
    {
        double probability = 0.5;
        if (argc > 3)
        {
            probability = atof(argv[3]);
        }
        generate_graph(graph, n, probability);
    }
    if (type == "fc" || type == "fs" || type == "fp")
    {
        if (argc < 4)
        {
            cout << "File path is required" << endl;
            return EXIT_FAILURE;
        }
        string file = argv[3];
        FILE *f = fopen(file.c_str(), "r");
        if (f == NULL)
        {
            cout << "File does not exist" << endl;
            return EXIT_FAILURE;
        }
        for (int i = 0; i < n; i++)
        {
            int x;
            for (int j = 0; j < n; j++)
            {
                fscanf(f, "%d", &x);
                graph[i][j] = x;
            }
        }
    }

    if (argc > 4)
    {
        level = atoi(argv[4]);
    }
    if (level < 1 || level > n - 3)
    {
        cout << "Level cannot be greater than n - 3 or less than 1" << endl;
        return EXIT_FAILURE;
    }

    if (argc > 5)
    {
        number_of_threads = atoi(argv[5]);
    }

    chrono::time_point<chrono::system_clock> start, end;
    if (type[0] == 'c' || type[1] == 'c')
    {
        start = std::chrono::system_clock::now();
        int result_s = find_hamiltonian_cycles_s(graph, n);
        end = std::chrono::system_clock::now();
        chrono::duration<double> time_s = end - start;

        int result_p = 0;
        chrono::time_point<chrono::system_clock> cuda_start, cuda_end;
        start = std::chrono::system_clock::now();
#ifdef __NVCC__
        int *cuda_result;
        cudaMalloc((void **)&cuda_result, sizeof(int) * number_of_threads);
        cudaMemset(cuda_result, 0, sizeof(int) * number_of_threads);

        bool **graph_arr;
        cudaMallocHost((void **)&graph_arr, sizeof(bool *) * n);
        for (int i = 0; i < n; i++)
        {
            cudaMallocHost((void **)&graph_arr[i], sizeof(bool) * n);
            cudaMemcpy(graph_arr[i], graph[i], sizeof(bool) * n, cudaMemcpyHostToHost);
        }
        bool **cuda_dev_arr;
        cudaMallocHost((void **)&cuda_dev_arr, sizeof(bool *) * n);
        for (int i = 0; i < n; i++)
        {
            cudaMalloc((void **)&cuda_dev_arr[i], sizeof(bool) * n);
            cudaMemcpy(cuda_dev_arr[i], graph_arr[i], sizeof(bool) * n, cudaMemcpyHostToDevice);
        }
        bool **cuda_graph;
        cudaMalloc((void **)&cuda_graph, sizeof(bool *) * n);
        cudaMemcpy(cuda_graph, cuda_dev_arr, sizeof(bool *) * n, cudaMemcpyHostToDevice);

        int *initial_path;
        cudaMallocHost((void **)&initial_path, sizeof(int) * n);
        for (int i = 0; i < n; i++)
        {
            initial_path[i] = i;
        }
        int **cuda_dev_paths_arr;
        cudaMallocHost((void **)&cuda_dev_paths_arr, sizeof(int *) * number_of_threads);
        for (int i = 0; i < number_of_threads; i++)
        {
            cudaMalloc((void **)&cuda_dev_paths_arr[i], sizeof(int) * n);
            cudaMemcpy(cuda_dev_paths_arr[i], initial_path, sizeof(int) * n, cudaMemcpyHostToDevice);
        }
        int **cuda_paths;
        cudaMalloc((void **)&cuda_paths, sizeof(int *) * number_of_threads);
        cudaMemcpy(cuda_paths, cuda_dev_paths_arr, sizeof(int *) * number_of_threads, cudaMemcpyHostToDevice);
        cuda_start = std::chrono::system_clock::now();
#endif

#ifdef __NVCC__
        find_hamiltonian_cycles_r<<<1, number_of_threads>>>(cuda_graph, cuda_paths, level, n, cuda_result, number_of_threads);
        cudaDeviceSynchronize();
#else
        result_p = find_hamiltonian_cycles_s(graph, n);
#endif

#ifdef __NVCC__
        cuda_end = std::chrono::system_clock::now();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Blad: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        int *result_arr;
        cudaMallocHost((void **)&result_arr, sizeof(int) * number_of_threads);
        cudaMemcpy(result_arr, cuda_result, sizeof(int) * number_of_threads, cudaMemcpyDeviceToHost);
        for (int i = 0; i < number_of_threads; i++)
        {
            result_p += result_arr[i];
        }
        cudaFreeHost(result_arr);
        cudaFree(cuda_result);

        for (int i = 0; i < n; i++)
        {
            cudaFree(cuda_dev_arr[i]);
        }
        cudaFree(cuda_graph);
        cudaFreeHost(cuda_dev_arr);
        for (int i = 0; i < n; i++)
        {
            cudaFreeHost(graph_arr[i]);
        }
        cudaFreeHost(graph_arr);
#endif
        end = std::chrono::system_clock::now();
        chrono::duration<double> time_p = end - start;
        free_graph(graph, n);
        if (result_s != result_p)
        {
            printf("Wrong results!\n");
            printf("Number of hamiltonian cycles in sequential: %d\n", result_s);
            printf("Number of hamiltonian cycles in parallel: %d\n", result_p);
        }
        else
        {
            printf("Time for sequential: %f\n", time_s.count());
            printf("Time for parallel: %f\n", time_p.count());
            printf("Number of hamiltonian cycles: %d\n", result_s);
        }
        // printf("%d,%f,%f\n", n, time_s.count(), time_p.count());
    }
    else if (type[0] == 's' || type[1] == 's')
    {
        start = std::chrono::system_clock::now();
        int result = find_hamiltonian_cycles_s(graph, n);
        end = std::chrono::system_clock::now();
        chrono::duration<double> time = end - start;
        free_graph(graph, n);
        printf("Time for sequential: %f\n", time.count());
        printf("Number of hamiltonian cycles: %d\n", result);
        // printf("%d,%f\n", n, time.count());
    }
    else
    {
        int result = 0;
        start = std::chrono::system_clock::now();
        chrono::time_point<chrono::system_clock> cuda_start, cuda_end;
#ifdef __NVCC__
        int *cuda_result;
        cudaMalloc((void **)&cuda_result, sizeof(int) * number_of_threads);
        cudaMemset(cuda_result, 0, sizeof(int) * number_of_threads);

        bool **graph_arr;
        cudaMallocHost((void **)&graph_arr, sizeof(bool *) * n);
        for (int i = 0; i < n; i++)
        {
            cudaMallocHost((void **)&graph_arr[i], sizeof(bool) * n);
            cudaMemcpy(graph_arr[i], graph[i], sizeof(bool) * n, cudaMemcpyHostToHost);
        }
        bool **cuda_dev_arr;
        cudaMallocHost((void **)&cuda_dev_arr, sizeof(bool *) * n);
        for (int i = 0; i < n; i++)
        {
            cudaMalloc((void **)&cuda_dev_arr[i], sizeof(bool) * n);
            cudaMemcpy(cuda_dev_arr[i], graph_arr[i], sizeof(bool) * n, cudaMemcpyHostToDevice);
        }
        bool **cuda_graph;
        cudaMalloc((void **)&cuda_graph, sizeof(bool *) * n);
        cudaMemcpy(cuda_graph, cuda_dev_arr, sizeof(bool *) * n, cudaMemcpyHostToDevice);

        int *initial_path;
        cudaMallocHost((void **)&initial_path, sizeof(int) * n);
        for (int i = 0; i < n; i++)
        {
            initial_path[i] = i;
        }
        int **cuda_dev_paths_arr;
        cudaMallocHost((void **)&cuda_dev_paths_arr, sizeof(int *) * number_of_threads);
        for (int i = 0; i < number_of_threads; i++)
        {
            cudaMalloc((void **)&cuda_dev_paths_arr[i], sizeof(int) * n);
            cudaMemcpy(cuda_dev_paths_arr[i], initial_path, sizeof(int) * n, cudaMemcpyHostToDevice);
        }
        int **cuda_paths;
        cudaMalloc((void **)&cuda_paths, sizeof(int *) * number_of_threads);
        cudaMemcpy(cuda_paths, cuda_dev_paths_arr, sizeof(int *) * number_of_threads, cudaMemcpyHostToDevice);
        cuda_start = std::chrono::system_clock::now();
#endif

#ifdef __NVCC__
        find_hamiltonian_cycles_r<<<1, number_of_threads>>>(cuda_graph, cuda_paths, level, n, cuda_result, number_of_threads);
        cudaDeviceSynchronize();
#else
        result = find_hamiltonian_cycles_s(graph, n);
#endif

#ifdef __NVCC__
        cuda_end = std::chrono::system_clock::now();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Blad: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        int *result_arr;
        cudaMallocHost((void **)&result_arr, sizeof(int) * number_of_threads);
        cudaMemcpy(result_arr, cuda_result, sizeof(int) * number_of_threads, cudaMemcpyDeviceToHost);
        for (int i = 0; i < number_of_threads; i++)
        {
            result += result_arr[i];
        }
        cudaFreeHost(result_arr);
        cudaFree(cuda_result);

        for (int i = 0; i < n; i++)
        {
            cudaFree(cuda_dev_arr[i]);
        }
        cudaFree(cuda_graph);
        cudaFreeHost(cuda_dev_arr);
        for (int i = 0; i < n; i++)
        {
            cudaFreeHost(graph_arr[i]);
        }
        cudaFreeHost(graph_arr);
#endif
        end = std::chrono::system_clock::now();
        chrono::duration<double> time = end - start;
        free_graph(graph, n);
        printf("Time for parallel: %f\n", time.count());
        chrono::duration<double> cuda_time = cuda_end - cuda_start;
        printf("Time for parallel kernel: %f\n", cuda_time.count());
        printf("Number of hamiltonian cycles: %d\n", result);
        // printf("%d,%d,%f\n", n, level, time.count());
    }

    return EXIT_SUCCESS;
}
