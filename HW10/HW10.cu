// Name: Nathanael Solagratia
// Robust Vector Dot product 
// nvcc HW10.cu -o temp
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU let do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.
 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values.
 4. In HW9 we had to do the final add "reduction' on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 256 // Threads in a block powers of 2

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
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

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
    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;
    
    GridSize.x = (N - 1) / BlockSize.x + 1; // This gives us the correct number of blocks.
    GridSize.y = 1;
    GridSize.z = 1;

    // Check if BlockSize.x is a power of 2
    if ((BlockSize.x & (BlockSize.x - 1)) != 0) {
        printf("Error: Block size is NOT a power of 2. Exiting...\n");
        exit(0);
    }
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.

__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
    int threadIndex = threadIdx.x;
    int vectorIndex = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float s_data[BLOCK_SIZE];

    // Each thread computes its partial product
    if (vectorIndex < n) {
        s_data[threadIndex] = a[vectorIndex] * b[vectorIndex];
    } else {
        s_data[threadIndex] = 0.0f;  // If the thread is out of bounds, set it to 0
    }

    __syncthreads();

    // Perform the folding (reduction) in shared memory within each block
    int fold = blockDim.x / 2;
    while (fold > 0) {
        if (threadIndex < fold) {
            s_data[threadIndex] += s_data[threadIndex + fold];
        }
        __syncthreads();
        fold /= 2;
    }

    // Only thread 0 in each block writes the result back to global memory
    if (threadIndex == 0) {
        atomicAdd(c, s_data[0]);
    }
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
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
	int deviceCount;
	cudaDeviceProp prop;
    cudaGetDeviceCount(&deviceCount);
	timeval start, end;
	long timeCPU, timeGPU;
	int P = (N/BLOCK_SIZE) * BLOCK_SIZE; // "Padding" size, the total threads that will run
	// To align the data structure's size to be an exact multiple of thread or block size, a memory access boundary, etc. 
    if (deviceCount == 0) {
       	printf("No CUDA devices found.");
        return 1;
    }
	printf("Found %d GPU(s)\n", deviceCount);

    int bestDevice = -1; // Default value just in case
    int minComputeCapabilityMajor = 3; // Minimum major compute capability
    int maxComputeCapabilityMinor = -1; // Default value for picking the better GPU based on the minor compute capability version

    for (int i = 0; i < deviceCount; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d Compute capability: %d.%d\n", i, prop.major, prop.minor);

        if (prop.major > minComputeCapabilityMajor || 
            (prop.major == minComputeCapabilityMajor && prop.minor > maxComputeCapabilityMinor)) {
            minComputeCapabilityMajor = prop.major; // Update the version of current GPU
            maxComputeCapabilityMinor = prop.minor;
            bestDevice = i; // Best device = Whatever current device is
        } // If there is a draw, simply picks the 1st device
    }

    if (bestDevice != -1) { // Most of the main function now is within the if statement
       	printf("\nSelected device with highest compute capability: Device %d \n", bestDevice);
        cudaSetDevice(bestDevice); // Built-in function for specified device
		printf("GPU model: %s\n", prop.name);
		// Setting up the GPU
		setUpDevices();
		int maxGridY = prop.maxGridSize[1]; // This is the maximum no. of blocks = 2^16
		// Check if no. blocks exceeded the limit
		printf("\nNo. of Blocks needed for %d vectors is %d blocks\nLimit of blocks %d", N, GridSize.x, maxGridY);
		printf("\nNo. of Threads that will run %d", P);
		if(GridSize.x > maxGridY) {
			printf("\nNo. of Blocks %d exceeded the limit of %d\nExiting ... ", GridSize.x, maxGridY);
			exit(0); // If so, exit
		}
		
		// Allocating the memory you will need.
		allocateMemory();
		
		// Ensure that every thread launched reads a valid memory location.
		// If not, the threads with index i >= N would attempt to read data past the end of the original array. 
		cudaMemset(A_GPU, 0, P * sizeof(float)); 
		cudaMemset(B_GPU, 0, P * sizeof(float));
		cudaMemset(C_GPU, 0, GridSize.x * sizeof(float)); // Because you'll only have "block size" amount of partial sums
		
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
		
		// Copy Memory from GPU to CPU	
		cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// Making sure the GPU and CPU wiat until each other are at the same place.
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
		
		DotGPU = 0.0;
		for(int i = 0; i < N; i += BlockSize.x)
		{
			DotGPU += C_CPU[i]; // C_GPU was copied into C_CPU. 
		}

		gettimeofday(&end, NULL);
		timeGPU = elaspedTime(start, end);
		
		// Checking to see if all went correctly.
		if (!check(DotCPU, DotGPU, Tolerance)) {
			printf("\n\n Something went wrong in the GPU dot product.\n");
			printf(" CPU Result = %f\n", DotCPU);
			printf(" GPU Result = %f\n", DotGPU);
			printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
			printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
		} else {
			printf("\n You did a dot product correctly on the GPU");
			printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
			printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
			printf("\n The CPU result was %f", DotCPU);
			printf("\n The GPU result was %f", DotGPU);
		}
		
		// Your done so cleanup your room.	
		CleanUp();	
		
		// Making sure it flushes out anything in the print buffer.
		printf("\n\n");
		
	} else {
        printf("No valid device");
        return 1;
    }

    return 0;
}

