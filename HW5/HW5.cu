// Name: Nathanael Solagratia
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPU(s) in this machine\n", count);
	
       for (int i=0; i < count; i++) {
	       cudaGetDeviceProperties(&prop, i);
	       cudaErrorCheck(__FILE__, __LINE__);

		   // GPU general info
	       printf("\n--- General Information for device %d ---\n", i);
	       printf("Name: %s\n", prop.name);
	       printf("  (The name of the GPU device)\n");
	       printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	       printf("  (CUDA compute capability version)\n");
	       printf("Clock rate: %.2f GHz\n", ((double)prop.clockRate)/1000000.0); // How fast the GPU can process instructions.
	       printf("  (Core clock frequency in Gigahertz)\n");
	       printf("Multiprocessor count: %d\n", prop.multiProcessorCount); // More SMs means more parallelism.
	       printf("  (Number of streaming multiprocessors (SMs))\n");
	       printf("Device copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled"); // If enabled, the GPU can transfer data and run kernels at the same time, improving performance.
	       printf("  (Can overlap data transfers and kernel execution)\n");
	       printf("Kernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled"); // On display GPUs, long kernels may be killed to keep the system responsive.
	       printf("  (Kernel execution timeout enabled on device)\n");
	       printf("Integrated: %s\n", prop.integrated ? "Yes" : "No"); // Almost all modern GPUs are discrete, especially Nvidia GPUs where all consumer GPUs are discrete
	       printf("  (Is the GPU integrated with the host)\n");
	       printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No"); // Whether the GPU can map host (CPU) memory into its address space.
	       printf("  (Can map host memory into CUDA address space)\n");
	       printf("Compute mode: %d\n", prop.computeMode); // Default (multiple processes can use the GPU).
	       printf("  (Device compute mode: 0=Default, 1=Exclusive, 2=Prohibited, 3=Exclusive Process)\n");
	       printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No"); // Whether the GPU can execute multiple kernels at the same time.
	       printf("  (Can execute multiple kernels concurrently)\n");
	       printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No"); // ECC helps detect and correct memory errors, important for reliability.
	       printf("  (Error Correction Code support)\n");
	       printf("PCI Bus ID: %d\n", prop.pciBusID);
	       printf("PCI Device ID: %d\n", prop.pciDeviceID); // Bus, Device, Domain IDs are for identifying and managing multiple GPUs.
	       printf("PCI Domain ID: %d\n", prop.pciDomainID);
	       printf("  (PCI location of the device)\n");
	       printf("L2 cache size: %d bytes\n", prop.l2CacheSize); // L2 cache helps speed up memory access for the GPU.
	       printf("  (Size of L2 cache in bytes)\n");
	       printf("Async engine count: %d\n", prop.asyncEngineCount); // More engines allow more concurrent operations.
	       printf("  (Number of asynchronous engines)\n");
	       printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No"); // Simplifies memory management in CUDA programs.
	       printf("  (Supports unified memory addressing)\n");
	       printf("Managed memory: %s\n", prop.managedMemory ? "Yes" : "No"); // Allows the system to automatically migrate data between host and GPU.
	       printf("  (Supports managed memory)\n");
	       printf("Is multi-GPU board: %s\n", prop.isMultiGpuBoard ? "Yes" : "No"); // Some boards have multiple GPUs on a single card.
	       printf("  (Is part of a multi-GPU board)\n");
	       printf("Multi-GPU board group ID: %d\n", prop.multiGpuBoardGroupID); // Used to identify GPUs that are on the same board.
	       printf("  (Group ID for multi-GPU boards)\n");

		   // GPU memory
	       printf("\n--- Memory Information for device %d ---\n", i);
	       printf("Total global memory: %ld bytes\n", prop.totalGlobalMem);
	       printf("  (Total amount of global memory on the device)\n");
	       printf("Total constant memory: %ld bytes\n", prop.totalConstMem); // Useful for storing values that do not change during kernel execution, like lookup tables.
	       printf("  (Total constant memory available)\n");
		   printf("Max mem pitch: %ld\n", prop.memPitch); // Useful for cudaMemcpy2D or cudaMemcpy3D functions, if more than the max value, the CUDA runtime will fail with an cudaErrorInvalidPitchValue error.
		   printf("  (Maximum pitch for memory copies)\n");
	       printf("Shared memory per block: %ld bytes\n", prop.sharedMemPerBlock); // Ideal for inter-thread communication and temporary storage within a block.
	       printf("  (Shared memory available per block)\n");
	       printf("Registers per block: %d\n", prop.regsPerBlock); // Fastest memory on the device, used for thread-local variables.
	       printf("  (32-bit registers available per block)\n");
	       printf("Memory clock rate: %.2f GHz\n", ((double)prop.memoryClockRate/1000000.0)); // Higher = faster memory access
	       printf("  (Memory clock frequency in Gigahertz)\n");
	       printf("Memory bus width: %d bits\n", prop.memoryBusWidth); // Higher = more data can be transferred per clock cycle
	       printf("  (Width of the memory bus in bits)\n");
	       printf("Texture alignment: %ld bytes\n", prop.textureAlignment); // Ensures efficient access when using textures in CUDA.
	       printf("  (Alignment requirement for textures)\n"); // cudaMallocArray, automatically handle the correct alignment

		   // GPU Thread, Block, and Kernel
	       printf("\n--- Thread/Block/Kernel Information for device %d ---\n", i);
	       printf("Warp size: %d\n", prop.warpSize); // A warp is a fundamental execution unit on NVIDIA GPUs, consisting of 32 threads (mostly)
	       printf("  (Number of threads in a warp)\n"); // These 32 threads within a warp, they all execute the same instruction at the same time.
	       printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); // The maximum number of threads you can launch in a single block.
	       printf("  (Maximum number of threads per block)\n");
	       printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); // The maximum size (in each dimension: x, y, z) for a block of threads.
	       printf("  (Maximum size of each block dimension)\n");
	       printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); // The maximum size (in each dimension: x, y, z) for the grid of blocks.
	       printf("  (Maximum size of each grid dimension)\n");
	       printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor); // The maximum number of threads that can be active on a single streaming multiprocessor at once.
	       printf("  (Maximum threads per multiprocessor)\n");

		   // GPU Texture for 2D/3D data
	       printf("\n--- Texture Information for device %d ---\n", i);
	       printf("Max 1D texture size: %d\n", prop.maxTexture1D); // This is the maximum number of elements 1D texture can hold.
	       printf("  (Maximum 1D texture size)\n");
	       printf("Max 2D texture size: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]); // 2D data for image processing, simulations, etc.
	       printf("  (Maximum 2D texture dimensions)\n");
	       printf("Max 3D texture size: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]); // 3D data for medical imaging (e.g., CT or MRI scans) or 3D simulations.
	       printf("  (Maximum 3D texture dimensions)\n");
       }
	return(0);
}

