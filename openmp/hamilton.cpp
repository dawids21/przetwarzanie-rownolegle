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

bool check_hamiltonian_cycle(vector<vector<bool>> &graph, vector<int> &path)
{
    for (int i = 0; i < path.size() - 1; i++)
    {
        if (graph[path[i]][path[i + 1]] == false)
        {
            return false;
        }
    }
    if (graph[path[path.size() - 1]][path[0]] == false)
    {
        return false;
    }
    return true;
}

// makes tests unreliable (results depend on the form of the graph)
bool check_partial_hamiltonian_cycle(vector<vector<bool>> &graph, vector<int> &path, int position)
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

int find_hamiltonian_cycles_s(vector<vector<bool>> &graph, vector<int> &path, int position)
{
    int n = graph.size();
    int cycles = 0;
    if (position == n - 2)
    {
        cycles += check_hamiltonian_cycle(graph, path);
        swap(path[n - 2], path[n - 1]);
        cycles += check_hamiltonian_cycle(graph, path);
        swap(path[n - 2], path[n - 1]);
        return cycles;
    }
    // if (!check_partial_hamiltonian_cycle(graph, path, position))
    // {
    //     return cycles;
    // }
    for (int i = position; i < n; i++)
    {
        if (i != position)
        {
            swap(path[position], path[i]);
        }
        cycles += find_hamiltonian_cycles_s(graph, path, position + 1);
        if (i != position)
        {
            swap(path[position], path[i]);
        }
    }
    return cycles;
}

int find_hamiltonian_cycles_s(vector<vector<bool>> &graph)
{
    int n = graph.size();
    vector<int> path(n);
    for (int i = 0; i < n; i++)
    {
        path[i] = i;
    }
    return find_hamiltonian_cycles_s(graph, path, 1);
}

int find_hamiltonian_cycles_r(vector<vector<bool>> &graph, vector<int> &path, int position, int level, int *acc)
{
    int n = graph.size();
    int cycles = 0;
    int thread_id = omp_get_thread_num();
    int num_of_threads = omp_get_num_threads();
    if (position == n - 2)
    {
        cycles += check_hamiltonian_cycle(graph, path);
        swap(path[n - 2], path[n - 1]);
        cycles += check_hamiltonian_cycle(graph, path);
        swap(path[n - 2], path[n - 1]);
        return cycles;
    }
    // if (!check_partial_hamiltonian_cycle(graph, path, position))
    // {
    //     return cycles;
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
                cycles += find_hamiltonian_cycles_r(graph, path, position + 1, level, acc);
            }
        }
        else
        {
            cycles += find_hamiltonian_cycles_r(graph, path, position + 1, level, acc);
        }
        if (i != position)
        {
            swap(path[position], path[i]);
        }
    }
    return cycles;
}

int find_hamiltonian_cycles_r(vector<vector<bool>> &graph, int level)
{
    int n = graph.size();
    vector<int> path(n);
    for (int i = 0; i < n; i++)
    {
        path[i] = i;
    }
    int acc = 0;
    return find_hamiltonian_cycles_r(graph, path, 1, level, &acc);
}

vector<vector<bool>> generate_graph(int n, double probability)
{
    srand(time(NULL));
    vector<vector<bool>> graph(n, vector<bool>(n));
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
    return graph;
}

int main(int argc, char const *argv[])
{
    string type = "c";
    int n = 10;
    int level = 1;
    vector<vector<bool>> graph;
    if (argc == 1) {
        cout << "Usage: " << argv[0] << " [type] [n] ([probability]|[file]) [level]" << endl
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
    if (type == "c" || type == "s" || type == "p")
    {
        double probability = 0.5;
        if (argc > 3)
        {
            probability = atof(argv[3]);
        }
        graph = generate_graph(n, probability);
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
        graph = vector<vector<bool>>(n, vector<bool>(n));
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

    chrono::time_point<chrono::system_clock> start, end;
    if (type[0] == 'c' || type[1] == 'c')
    {
        start = std::chrono::system_clock::now();
        int result_s = find_hamiltonian_cycles_s(graph);
        end = std::chrono::system_clock::now();
        chrono::duration<double> time_s = end - start;

        start = std::chrono::system_clock::now();
        int result_p = 0;
#pragma omp parallel reduction(+ : result_p) if (n > 2)
        {
            result_p += find_hamiltonian_cycles_r(graph, level);
        }
        end = std::chrono::system_clock::now();
        chrono::duration<double> time_p = end - start;
        assert(result_s == result_p);
        printf("Time for sequential: %f\n", time_s.count());
        printf("Time for parallel: %f\n", time_p.count());
        printf("Number of hamiltonian cycles: %d\n", result_s);
        // printf("%d,%f,%f\n", n, time_s.count(), time_p.count());
    }
    else if (type[0] == 's' || type[1] == 's')
    {
        start = std::chrono::system_clock::now();
        int result = find_hamiltonian_cycles_s(graph);
        end = std::chrono::system_clock::now();
        chrono::duration<double> time = end - start;
        printf("Time for sequential: %f\n", time.count());
        printf("Number of hamiltonian cycles: %d\n", result);
        // printf("%d,%f\n", n, time.count());
    }
    else
    {
        int result = 0;
        start = std::chrono::system_clock::now();
#pragma omp parallel reduction(+ : result) if (n > 2)
        {
            result += find_hamiltonian_cycles_r(graph, level);
        }
        end = std::chrono::system_clock::now();
        chrono::duration<double> time = end - start;
        printf("Time for parallel: %f\n", time.count());
        printf("Number of hamiltonian cycles: %d\n", result);
        // printf("%d,%d,%f\n", n, level, time.count());
    }

    return EXIT_SUCCESS;
}

