#ifdef __NVCC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#endif

#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <complex>
#include <fstream>

#define MIN_X -2
#define MAX_X 2
#define MIN_Y -2
#define MAX_Y 2
#define MAX_COLOR 255

using namespace std;

void generate_ppm(const int *pixels, int width, int height, string filename)
{
    ofstream ppm_file(filename);
    if (!ppm_file)
    {
        cerr << "Failed to open the PPM file." << endl;
        return;
    }

    ppm_file << "P3" << endl;
    ppm_file << width << " " << height << " 255" << endl;

    for (int i = 0; i < width * height; ++i)
    {
        ppm_file << "0 0 " << pixels[i] << endl;
    }

    ppm_file.close();
}

int calculate_mandelbrot_s(complex<float> c, int max_iteration)
{
    complex<float> z = 0;
    int iteration = 0;
    while (abs(z) < 2 && iteration < max_iteration)
    {
        z = z * z + c;
        iteration++;
    }
    return iteration;
}

void calculate_mandelbrot_set_s(int width, int height, int max_iteration, int *mandelbrot_set)
{
    float dx = (float)(MAX_X - MIN_X) / width;
    float dy = (float)(MAX_Y - MIN_Y) / height;
    for (int i = 0; i < width; i++)
    {
        float x = (float)MIN_X + i * dx;
        for (int j = 0; j < height; j++)
        {
            float y = (float)MIN_Y + j * dy;
            complex<float> c(x, y);
            int iterations = calculate_mandelbrot_s(c, max_iteration);
            mandelbrot_set[j * height + i] = MAX_COLOR - (MAX_COLOR * iterations / max_iteration);
        }
    }
}

#ifdef __NVCC__
__device__ int calculate_mandelbrot_r(cuComplex c, int max_iteration)
{
    cuComplex z = make_cuComplex(0, 0);
    int iteration = 0;
    while (cuCabsf(z) < 2 && iteration < max_iteration)
    {
        z = cuCaddf(cuCmulf(z, z), c);
        iteration++;
    }
    return iteration;
}

__global__ void calculate_mandelbrot_set_r(int width, int height, int max_iteration, int *mandelbrot_set)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        float x = (float)MIN_X + (float)i * (MAX_X - MIN_X) / width;
        float y = (float)MIN_Y + (float)j * (MAX_Y - MIN_Y) / height;
        cuComplex c = make_cuComplex(x, y);
        int iterations = calculate_mandelbrot_r(c, max_iteration);
        mandelbrot_set[j * height + i] = MAX_COLOR - (MAX_COLOR * iterations / max_iteration);
    }
}
#endif

int main(int argc, char const *argv[])
{
    string type = "c";
    int width = 400;
    int height = 400;
    int max_iteration = 100;
    string filename = "";
    if (argc == 1)
    {
        cout << "Usage: " << argv[0] << " [type] [width] [height] [max_iteration] [filename]" << endl
             << endl;
        cout << "type:" << endl;
        cout << "c - compare results, s - serial, p - parallel" << endl
             << endl;
        cout << "width: width of image" << endl;
        cout << "height: height of image" << endl
             << endl;
        cout << "max_iteration: number of iterations" << endl
             << endl;
        cout << "filename: name of output file" << endl
             << endl;
        return EXIT_FAILURE;
    }
    if (argc > 1)
    {
        type = argv[1];
    }
    if (type != "c" && type != "s" && type != "p")
    {
        cout << "Type must be c, s, p" << endl;
        return EXIT_FAILURE;
    }
    if (argc > 2)
    {
        width = atoi(argv[2]);
    }
    if (argc > 3)
    {
        height = atoi(argv[3]);
    }
    if (argc > 4)
    {
        max_iteration = atoi(argv[4]);
    }
    if (argc > 5)
    {
        filename = argv[5];
    }

    chrono::time_point<chrono::system_clock> start, end;
    if (type == "c")
    {
        int *mandelbrot_set_s = (int *)malloc(sizeof(int) * width * height);
        start = std::chrono::system_clock::now();
        calculate_mandelbrot_set_s(width, height, max_iteration, mandelbrot_set_s);
        end = std::chrono::system_clock::now();
        chrono::duration<double> time_s = end - start;

        int *mandelbrot_set_p = (int *)malloc(sizeof(int) * width * height);
        start = std::chrono::system_clock::now();
#ifdef __NVCC__
        int *cuda_mandelbrot_set;
        cudaMalloc((void **)&cuda_mandelbrot_set, sizeof(int) * width * height);
        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
#endif

#ifdef __NVCC__
        calculate_mandelbrot_set_r<<<gridSize, blockSize>>>(width, height, max_iteration, cuda_mandelbrot_set);
        cudaDeviceSynchronize();
#else
        calculate_mandelbrot_set_s(width, height, max_iteration, mandelbrot_set_p);
#endif

#ifdef __NVCC__
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        int *host_mandelbrot_set;
        cudaMallocHost((void **)&host_mandelbrot_set, sizeof(int) * width * height);
        cudaMemcpy(host_mandelbrot_set, cuda_mandelbrot_set, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
        memcpy(mandelbrot_set_p, host_mandelbrot_set, sizeof(int) * width * height);
        cudaFreeHost(host_mandelbrot_set);
        cudaFree(cuda_mandelbrot_set);
#endif
        end = std::chrono::system_clock::now();
        chrono::duration<double> time_p = end - start;
        printf("Time for sequential: %f\n", time_s.count());
        printf("Time for parallel: %f\n", time_p.count());
        if (filename != "")
        {
            generate_ppm(mandelbrot_set_p, width, height, filename);
        }
        free(mandelbrot_set_s);
        free(mandelbrot_set_p);
    }
    else if (type[0] == 's' || type[1] == 's')
    {
        int *mandelbrot_set = (int *)malloc(sizeof(int) * width * height);
        start = std::chrono::system_clock::now();
        calculate_mandelbrot_set_s(width, height, max_iteration, mandelbrot_set);
        end = std::chrono::system_clock::now();
        chrono::duration<double> time = end - start;
        printf("Time for sequential: %f\n", time.count());
        if (filename != "")
        {
            generate_ppm(mandelbrot_set, width, height, filename);
        }
        free(mandelbrot_set);
    }
    else
    {
        int *mandelbrot_set = (int *)malloc(sizeof(int) * width * height);
        start = std::chrono::system_clock::now();
#ifdef __NVCC__
        int *cuda_mandelbrot_set;
        cudaMalloc((void **)&cuda_mandelbrot_set, sizeof(int) * width * height);
        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
#endif

#ifdef __NVCC__
        calculate_mandelbrot_set_r<<<gridSize, blockSize>>>(width, height, max_iteration, cuda_mandelbrot_set);
        cudaDeviceSynchronize();
#else
        calculate_mandelbrot_set_s(width, height, max_iteration, mandelbrot_set);
#endif

#ifdef __NVCC__
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        int *host_mandelbrot_set;
        cudaMallocHost((void **)&host_mandelbrot_set, sizeof(int) * width * height);
        cudaMemcpy(host_mandelbrot_set, cuda_mandelbrot_set, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
        memcpy(mandelbrot_set, host_mandelbrot_set, sizeof(int) * width * height);
        cudaFreeHost(host_mandelbrot_set);
        cudaFree(cuda_mandelbrot_set);
#endif
        end = std::chrono::system_clock::now();
        chrono::duration<double> time = end - start;
        printf("Time for parallel: %f\n", time.count());
        if (filename != "")
        {
            generate_ppm(mandelbrot_set, width, height, filename);
        }
        free(mandelbrot_set);
    }

    return EXIT_SUCCESS;
}