// Name: Nathanael Solagratia
// Page-locked memory test
// nvcc 13PageLockedMemory.cu -o t

/*
 What to do:
 Read about **page-locked (pinned) memory**. Fill in the ???s in this code to understand how to
 set up and test page-locked memory on the host.
*/

/*
 Purpose:
 To learn how page-locked (pinned) memory works and how to use it effectively.
 
 Page-locked memory guarantees that it will never page this memory out to disk ensuring to stay in place
 in the physical memory. In other words, become safe for the OS to allow application access to the physical addresses
 of the memory, since the memory will not be relocated.

 If we use pageable memory like malloc, the copy happens twice because the CUDA driver still uses
 direct memory access to transfer the memory to the GPU. Therefore, its slower.

 But we can't really use this all the time because virtual memory is gone, then the computer running the
 application must have available physical memory for every page-locked memory. In other words, this consumes
 more memory and we will ran out of memory faster.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define SIZE 2000000 
#define NUMBER_OF_COPIES 1000

//Globals
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
cudaEvent_t StartEvent, StopEvent;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void cleanUp();
void copyPageableMemoryUp();
void copyPageLockedMemoryUp();
void copyPageableMemoryDown();
void copyPageLockedMemoryDown();

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
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&NumbersOnGPU, SIZE*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate pageable Host (CPU) Memory
	PageableNumbersOnCPU = (float*)malloc(SIZE*sizeof(float));
	
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc( (void**)&PageableNumbersOnCPU, SIZE * sizeof(float), cudaHostAllocDefault );
	cudaErrorCheck(__FILE__, __LINE__);
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(NumbersOnGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(PageableNumbersOnCPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// free(PageableNumbersOnCPU); 
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

void copyPageableMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(NumbersOnGPU, PageableNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageableMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy( NumbersOnGPU, PageableNumbersOnCPU, SIZE * sizeof(float), cudaMemcpyHostToDevice );
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory up = %3.1f milliseconds", timeEvent);
	printf("\nINFO: copyPageableMemoryUp() events recorded successfully.");
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory up = %3.1f milliseconds", timeEvent);
	printf("\nINFO: copyPageLockedMemoryUp() events recorded successfully.");
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory down = %3.1f milliseconds", timeEvent);
	printf("\nINFO: copyPageableMemoryDown() events recorded successfully.");
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory down = %3.1f milliseconds", timeEvent);
	printf("\nINFO: copyPageLockedMemoryDown() events recorded successfully.");
	
	printf("\n\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
