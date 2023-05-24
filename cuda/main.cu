#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

int gcd(int a, int b)
{
	int t = 1;
	while (t)
	{
		t = a % b;
		a = b;
		b = t;
	}
	return a;
}

__global__ void kernelgcd(int n, int size, int r, int numOfThreads, int liczba, int *dane, int *wynik)
{
	int tid = threadIdx.x;
	for (int i = tid * size; i < (tid + 1) * size && i < n; i++)
	{
		int a = dane[i];
		int b = liczba;
		int temp = 1;
		while (temp)
		{
			temp = a % b;
			a = b;
			b = temp;
		}
		wynik[i] = a;
	}
	if (tid == numOfThreads - 1)
	{
		for (int i = (tid + 1) * size; i < (tid + 1) * size + r && i < n; i++)
		{
			int a = dane[i];
			int b = liczba;
			int temp = 1;
			while (temp)
			{
				temp = a % b;
				a = b;
				b = temp;
			}
			wynik[i] = a;
		}
	}
}

int main(int argc, char *argv[])
{
	int i;
	int *dane;
	int *cuda_dane;
	int *wynik1;
	int *wynik2;

	int *cuda_wynik;
	int liczba = 24;
	int n = 1025;
	int numOfThreads = 1024;
	cudaError_t err;

	if (argc > 1)
		n = atoi(argv[1]);
	else
		n = 10;
	if (argc > 2)
		liczba = atoi(argv[2]);
	else
		liczba = 24;

	if (n < numOfThreads)
	{
		numOfThreads = n;
	}

	dane = (int *)malloc(n * sizeof(int));
	wynik1 = (int *)malloc(n * sizeof(int));
	wynik2 = (int *)malloc(n * sizeof(int));

	for (i = 0; i < n; i++)
		dane[i] = rand() % 1000;

	cudaMalloc((void **)&cuda_dane, sizeof(int) * n);
	cudaMalloc((void **)&cuda_wynik, sizeof(int) * n);

	for (i = 0; i < n; i++)
	{
		wynik1[i] = gcd(dane[i], liczba);
	}

	int r = n % numOfThreads;
	int size;
	if (r == 0)
	{
		size = n / numOfThreads;
	}
	else
	{
		size = (n - r) / numOfThreads;
	}

	cudaMemcpy(cuda_dane, dane, sizeof(int) * n, cudaMemcpyHostToDevice);
	kernelgcd<<<1, numOfThreads>>>(n, size, r, numOfThreads, liczba, cuda_dane, cuda_wynik);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Blad: %s\n", cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	cudaMemcpy(wynik2, cuda_wynik, sizeof(int) * n, cudaMemcpyDeviceToHost);

	for (i = 0; i < n; i++)
	{
		if (wynik1[i] != wynik2[i])
		{
			printf("gcd(%d,%d)=CPU:%d vs. GPU:%d\n", liczba, dane[i], wynik1[i], wynik2[i]);
		}
	}

	cudaFree(cuda_wynik);
	cudaFree(cuda_dane);

	free(dane);
	free(wynik1);
	free(wynik2);
	return EXIT_SUCCESS;
}
