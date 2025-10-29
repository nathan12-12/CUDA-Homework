// Name: Nathan S.
// Two streams overlapped smartly 
// nvcc O_SmartStreams.cu -o temp
/*
 What to do:
 Read about CUDA stream cooperation.

This code provides most of the setup needed to create two CUDA streams. 
Complete the implementation by replacing all the ???s.

Once the two streams are working, overlap them in a smart way to improve performance.
*/

/*
 Purpose:
 To learn how to use CUDA streams intelligently.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define DATA_CHUNKS (1024*1024) 
#define ENTIRE_DATA_SET (20*DATA_CHUNKS)
#define MAX_RANDOM_NUMBER 1000
#define BLOCK_SIZE 256
#define TOLERANCE 1e-4 

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A0_GPU, *B0_GPU, *C0_GPU, *A1_GPU, *B1_GPU, *C1_GPU; //GPU pointers
float *C_CPU_Test; // For testing purposes
cudaEvent_t StartEvent, StopEvent;
cudaStream_t Stream0, Stream1;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

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
	
	cudaDeviceProp prop;
	int whichDevice;
	
	cudaGetDevice(&whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1)
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	cudaStreamCreate(&Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamCreate(&Stream1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);

	// For testing purposes regular malloc
	C_CPU_Test = (float*)malloc(ENTIRE_DATA_SET*sizeof(float));
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++)
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(A1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(A_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaStreamDestroy(Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamDestroy(Stream1);
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n)
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	loadData();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS*2)
	{
		cudaMemset(C0_GPU, 0, DATA_CHUNKS * sizeof(float));
		cudaMemset(C1_GPU, 0, DATA_CHUNKS * sizeof(float));

		// With "&" like &A_CPU[i] you can get the address of the i-th element of the array.
		// With A_CPU+i does the same thing, but its called pointer arithmetic.
		// 1st Stream
		cudaMemcpyAsync(A0_GPU, &A_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B0_GPU, &B_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);

		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(A0_GPU, B0_GPU, C0_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpyAsync(&C_CPU[i], C0_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// 2nd Stream
		cudaMemcpyAsync(A1_GPU, &A_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B1_GPU, &B_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);

		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream1>>>(A1_GPU, B1_GPU, C1_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);

		// Factor in the offset for the 2nd half of the data chunk
		cudaMemcpyAsync(&C_CPU[i + DATA_CHUNKS], C1_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	// Make the CPU wait until the Streams have finishd before it continues.
	cudaStreamSynchronize(Stream0);
	cudaStreamSynchronize(Stream1);
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	printf("\nFinal Results from GPU, the first 10 elements:\n");
	for(int i = 0; i < 10; i++) {
		printf("C_CPU[%d] = %f\n", i, C_CPU[i]);
	}
	for(int i = 0; i < ENTIRE_DATA_SET; i++) {
		C_CPU_Test[i] = sin(A_CPU[i]) + cos(B_CPU[i]);
	}
	bool allMatch = false;
	for(int i = 0; i < ENTIRE_DATA_SET; i++) {// Check all elements
		if(fabs(C_CPU[i] - C_CPU_Test[i]) > TOLERANCE)
			printf("Screw up in %d, answer: %f, expected: %f\n", i, C_CPU[i], C_CPU_Test[i]);
		else
			allMatch = true;
	}
	if (allMatch)
		printf("\nCPU and GPU results match within tolerance of %e\n", TOLERANCE);

	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
