// Name: Nathanael Solagratia
// Histogram useing atomics in global memory and shared memory.
// nvcc 12HistogramUseingAtomics.cu -o t

// Include files
#include <sys/time.h>
#include <stdio.h>

/*
 Note: The Max int value is 2,147,483,647
 The length of the sequence of random number that srand generates is 2^32,
 that is 4,294,967,296 which is bigger than the largest int but the max for an unsigned int.
*/

// Defines
#define NUMBER_OF_RANDOM_NUMBERS 2147483
#define NUMBER_OF_BINS 10
#define MAX_RANDOM_NUMBER 100.0f

// Global variables
float *RandomNumbersGPU;
int *HistogramGPU;
float *RandomNumbersCPU;
int *HistogramCPU;
int *HistogramCPUTemp; // Use it to hod the GPU histogram past back so we can compair to CPU histogram.
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//Function prototypes
void cudaErrorCheck(const char *, int);
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
void fillHistogramCPU();
__global__ void fillHistogramGPU(float *, int *);
int main();

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

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaErrorCheck(__FILE__, __LINE__);

	//BlockSize.x = 222;
	BlockSize.x = 2*(prop.multiProcessorCount); // - Set the block size to **twice** the number of multiprocessors on the GPU.
	printf("\nMultiprocessor count = %d\nBlockSize.x = %d\n", prop.multiProcessorCount, BlockSize.x);
	if(prop.maxThreadsDim[0] < BlockSize.x)
	{
		printf("\n You are trying to create more threads (%d) than your GPU can support on a block (%d).\n Good Bye\n", BlockSize.x, prop.maxThreadsDim[0]);
		exit(0);
	}
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (NUMBER_OF_RANDOM_NUMBERS - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	if(prop.maxGridSize[0] < GridSize.x)
	{
		printf("\n You are trying to create more blocks (%d) than your GPU can suppport (%d).\n Good Bye\n", GridSize.x, prop.maxGridSize[0]);
		exit(0);
	}
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&RandomNumbersGPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&HistogramGPU, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	RandomNumbersCPU = (float*)malloc(NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	HistogramCPU = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	HistogramCPUTemp = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	
	//Setting the the histograms to zero.
	cudaMemset(HistogramGPU, 0, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
	memset(HistogramCPU, 0, NUMBER_OF_BINS*sizeof(int));
}

//Loading random numbers.
void Innitialize()
{
	time_t t;
	srand((unsigned) time(&t));
	
	// rand() returns an int in [0, RAND_MAX] "end points included".
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{		
		RandomNumbersCPU[i] = MAX_RANDOM_NUMBER*(float)rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	cudaFree(RandomNumbersGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	free(RandomNumbersCPU); 
	free(HistogramCPU);
	free(HistogramCPUTemp);
	//printf("\n Cleanup Done.");
}

void fillHistogramCPU()
{
	float breakPoint;
	int k, done;
	float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{
		breakPoint = stepSize;
		k = 0;
		done =0;
		while(done == 0)
		{
			if(RandomNumbersCPU[i] < breakPoint)
			{
				HistogramCPU[k]++; 
				done = 1;
			}
			
			if(NUMBER_OF_BINS < k)
			{
				printf("\n k is too big\n");
				exit(0);
			}
			k++;
			breakPoint += stepSize;
		}
	}
}

/*
 What to do:
 This code generates a series of random numbers and places them into bins based on size ranges using the CPU.

 Your task:
 - Create a binning scheme that utilizes the GPU. Done.
 - Take advantage of both global and shared memory atomic operations. Done
 - The function call has already been provided. Ok
 - Set the block size to **twice** the number of multiprocessors on the GPU. Done
*/

/*
 Purpose:
 To learn how to use atomic operations at both the shared and global memory levels.
 Along the way, you'll also learn a bit about generating random numbers using `srand`,
 which will come in handy when we use `curand` in a later assignment.

The problem with only using Global memory, is the fact that thousands of threads were competing for access
to a relatively small number of memory addresses. To address this issue we need to split our histogram
computation into two phases.

1st. Each parallel block will compute a separate histogram of the data and since each block does this independently,
we can compute these histograms in shared memory. The fact that only NUMBER_OF_BINS (10) threads will now competing for
twice the size of multiprocessor (48) accesses.

2nd. Merge each block's temporary histogram into the global memory. Adding each entry in the (local) block's histogram
to the corresponding global histogram.
*/ 
// This is the kernel. It is the function that will run on the GPU.
__global__ void fillHistogramGPU(float *randomNumbers, int *hist)
{
	// Declare Shared Memory for Local Histogram
    __shared__ unsigned int temp[NUMBER_OF_BINS];

    // Initialize Shared Memory to 0
    temp[threadIdx.x] = 0;

    // Synchronize to ensure all threads see the zeroed shared memory
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

    while (i < NUMBER_OF_RANDOM_NUMBERS) {
		// Correct binning calculation
		// Example if randomNumbers[i] = 45 ---> int(45/10) ---> 4 ---> add to bin 4
        int where = (int)(randomNumbers[i]/NUMBER_OF_BINS); // Typecast to int
        if (where >= 0 && where < NUMBER_OF_BINS) { // where should be between 0 and 9
            // Use atomicAdd on the shared memory location
            atomicAdd(&(temp[where]), 1);
        }
		i += offset;
    }

    // Synchronize to ensure all local updates are complete
    __syncthreads();

    // Use atomicAdd again on the Global memory
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
	// So after that hullabaloo, our GPU histogram computation is faster than the CPU version
}


int main()
{
	float time;
	timeval start, end;
	
	long int test = NUMBER_OF_RANDOM_NUMBERS;
	if(2147483647 < test)
	{
		printf("\nThe length of your vector is longer than the largest integer value allowed of 2,147,483,647.\n");
		printf("You should check your code.\n Good Bye\n");
		exit(0);
	}
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using and padding with zero vector will be a factor of block size.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	gettimeofday(&start, NULL);
	fillHistogramCPU();
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime on CPU = %.15f milliseconds\n", (time/1000.0));
	float time_cpu = time;
	
	gettimeofday(&start, NULL);
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(RandomNumbersGPU, RandomNumbersCPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	fillHistogramGPU<<<GridSize,BlockSize>>>(RandomNumbersGPU, HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(HistogramCPUTemp, HistogramGPU, NUMBER_OF_BINS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime on GPU = %.15f milliseconds\n", (time/1000.0));
	float time_gpu = time;
	printf("\nGPU is = %.15f%% faster\n", ((time_cpu/time_gpu)-1)*100);
	
	//Check
	for(int i = 0; i < NUMBER_OF_BINS; i++)
	{
		printf("\nDeference in histogram bins %d is %d.", i, abs(HistogramCPUTemp[i] - HistogramCPU[i]));
	}
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	printf("\n\n");
	return(0);
}
