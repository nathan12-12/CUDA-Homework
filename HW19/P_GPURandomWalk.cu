// Name: Nathan S.
// GPU random walk. 
// nvcc P_GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

// Defines
#define NUM_WALKS 20
#define WALK_STEPS 1000

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid

// Checking CUDA error codes
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line-1);
		exit(0);
	}
}

void setUpDevices()
{
	BlockSize.x = 256; // Optimal block size
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = ((NUM_WALKS - 1)/BlockSize.x) + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
}

// Kernel for performing a 2D random walk
__global__ void randomWalkKernel(int* final_positionX, int* final_positionY, unsigned long long base_seed) {
    // Each thread represents one random walk
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_WALKS) {
        // Initialize cuRAND state for a random walk
        curandState_t state;
        
        // Use the base_seed combined with the thread ID (as the sequence number) 
        // to ensure a different, reproducible sequence for each walk.
        curand_init(base_seed, tid, 0, &state);

        int positionX = 0;
        int positionY = 0;

        // Perform the random walk (in 2D, x & y plane)
        for (int i = 0; i < WALK_STEPS; ++i) {
            // Generate two uniform floats in [0, 1) for X and Y direction
            float rand_x = curand_uniform(&state);
            float rand_y = curand_uniform(&state);

            // Step left (-1) or right (+1) based on comparison with 0.5
			// "?" is a shorthand for if-then-else, if less than 0.5 go left (-1), else go right (+1)
            positionX += (rand_x < 0.5f) ? -1 : 1; 
            positionY += (rand_y < 0.5f) ? -1 : 1;
        }

        // Store the final positions in Unified Memory
        final_positionX[tid] = positionX;
        final_positionY[tid] = positionY;
    }
}

int main() {
	// Setting up the GPU
	setUpDevices();
	
    // Intiate values for final position x and y variables
    int *d_final_positionX = NULL;
    int *d_final_positionY = NULL;
    size_t mem_size = NUM_WALKS * sizeof(int);
    
    // Allocates unified memory accessible by both CPU and GPU
    cudaMallocManaged((void**)&d_final_positionX, mem_size);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged((void**)&d_final_positionY, mem_size);
	cudaErrorCheck(__FILE__, __LINE__);

    // Get a time-based seed for the random walks initialization
    unsigned long long base_seed = (unsigned long long)time(NULL);

    printf("Starting %d random walks (Steps: %d) on the GPU\n", NUM_WALKS, WALK_STEPS);
    printf("Using base seed: %llu\n", base_seed);

    randomWalkKernel<<<GridSize.x, BlockSize.x>>>(d_final_positionX, d_final_positionY, base_seed);
    
    // Wait for the GPU to finish all computation and data migration
    cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

    // Print out all 20 final positions (accessible directly from the CPU via Unified Memory)
    printf("\nFinal Positions\n");
    for (int i = 0; i < NUM_WALKS; i++) {
        // The seed for walk 'i' is derived from the base_seed and thread ID 'i'.
        printf("Walk %2d (Seed: %llu): Final Position = (%d, %d)\n", i+1, base_seed + i, d_final_positionX[i], d_final_positionY[i]); 
	}
	printf("\nCleaning Unified Memory...\n\n");

    // Clean up Unified Memory
	cudaFree(d_final_positionX);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(d_final_positionY);
	cudaErrorCheck(__FILE__, __LINE__);

    return 0;

}
