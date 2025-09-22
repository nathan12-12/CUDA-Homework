// Name: Nathanael Solagratia
// Vector Dot product on 1 block 
// nvcc HW8.cu -o temp
/*
 What to do:
 This code uses the CPU to compute the dot product of two vectors of length N. 
 It includes a skeleton for setting up a GPU dot product, but that part is currently empty.
 Additionally, the CPU code is somewhat convoluted, but it is structured this way to parallel 
 the GPU code you will need to write. The program will also verify whether you have correctly 
 implemented the dot product on the GPU.
 Leave the block and vector sizes as:
 Block = 1000
 N = 823
 Use folding at the block level when you do the addition reduction step.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

// Defines
#define N 823 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, float*, int);  // Fixed function signature
__global__ void dotProductGPU(float*, float*, float*, int); // Fixed function signature
bool check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 1000;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // Correct no. of blocks needed to cover N elements
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(GridSize.x*sizeof(float));  // Fixed: allocate space for partial results
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,GridSize.x*sizeof(float));  // Fixed: allocate space for partial results
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Computing dot product on the CPU - Fixed function
void dotProductCPU(float *a, float *b, float *c, int n)
{
	// First compute element-wise products
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] * b[id];
	}
	
	// Then sum all products into c[0]
	for(int id = 1; id < n; id++)
	{ 
		c[0] += c[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
    __shared__ float cache[1000]; // The __shared__ keyword declares an array cache in shared memory, accessible by all threads within a block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;

    // Each thread calculates its unique index tid based on its threadIdx and blockIdx. The temp variable accumulates the dot product result for each thread.
    while(tid < n) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

    // Store result in shared memory for later reduction
    cache[cacheIndex] = temp;
    __syncthreads(); // This function ensures everyone (the cache) finishes before moving on

    // Reduction step, combining elements of an array in parallel
	// To sum up the partial results of each thread within a block
    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads(); // This function ensures everyone (the cache) finishes before moving on
        i /= 2;
    }

    // Write the final reduced result of the block to the global output array
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf", percentError);
	printf("\n CPU result = %f", cpuAnswer);
	printf("\n GPU result = %f", gpuAnswer);
	
	if(percentError < tolerence) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Synchronizing first before copying memory.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost);  // Copy correct amount
	// C_GPU only needs space for GridSize.x partial results, not N results. The rest would be uninitialized memory, leading to incorrect results.
	cudaErrorCheck(__FILE__, __LINE__);

	DotGPU = 0;
	for(int i = 0; i < GridSize.x; i++)
		DotGPU += C_CPU[i]; // Sum up partial results from each block

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}