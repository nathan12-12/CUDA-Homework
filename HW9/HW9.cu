// Name: Nathanael Solagratia
// Vector Dot product on many block and useing shared memory
// nvcc HW9.cu -o temp && ./temp 25000
// If its so big it will cause segmentation fault
/*
 What to do:
 This code is the solution to HW8. It finds the dot product of vectors that are smaller than the block size.
 Extend this code so that it sets as many blocks as needed for a set thread count and vector length.
 Use shared memory in your blocks to speed up your code.
 You will have to do the final reduction on the CPU.
 Set your thread count to 200 (block size = 200). Set N to different values to check your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
//#define N 250000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; // CPU pointers
float *A_GPU, *B_GPU, *C_GPU; // GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; // This variable will hold the Dimensions of your blocks
dim3 GridSize;  // This variable will hold the Dimensions of your grid
float Tolerance = 0.01;
int THREADS_PER_BLOCK = 200; // Any block size

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices(int);
void allocateMemory(int);
void innitialize(int);
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

void cudaErrorCheck(const char *file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
        exit(0);
    }
}

void setUpDevices(int N) {
    BlockSize.x = THREADS_PER_BLOCK;
    BlockSize.y = 1;
    BlockSize.z = 1;
    
    GridSize.x = (N - 1) / BlockSize.x + 1;
    GridSize.y = 1;
    GridSize.z = 1;
}

void allocateMemory(int N) {    
    cudaHostAlloc(&A_CPU, N * sizeof(float), cudaHostAllocDefault);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaHostAlloc(&B_CPU, N * sizeof(float), cudaHostAllocDefault);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaHostAlloc(&C_CPU, GridSize.x * sizeof(float), cudaHostAllocDefault);
    cudaErrorCheck(__FILE__, __LINE__);
    
    cudaMalloc(&A_GPU, N * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU, N * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU, GridSize.x * sizeof(float)); 
    cudaErrorCheck(__FILE__, __LINE__);}

void innitialize(int N) {
    for (int i = 0; i < N; i++) {        
        A_CPU[i] = (float)i;    
        B_CPU[i] = (float)(3 * i);
    }
}

void dotProductCPU(float *a, float *b, int n) {
    for (int id = 0; id < n; id++) { 
        C_CPU[id] = a[id] * b[id];
    }
    
    for (int id = 1; id < n; id++) { 
        C_CPU[0] += C_CPU[id];
    }
}

__global__ void dotProductGPU(float* a, float* b, float* c, int n) {
	// "extern" means the size of shared memory array is not fixed at compile time but will be determined dynamically at kernel launch time.
    extern __shared__ float shared_mem[]; // Dynamically allocated shared memory array for the block.
	// "tid" maps the thread to a unique global index across all threads in the grid, used to access elements of a and b.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// "shared_tid" used for storing and accessing partial sums during the reduction phase.
    int shared_tid = threadIdx.x;

    // Compute partial dot product for this thread
    float sum = 0.0f;
	// "stride" ensures that all elements are processed even if n > total number of threads.
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { // Jumping by "stride" ensures all elements are covered.
        sum += a[i] * b[i];
    }
	// Each thread stores its partial sum in shared memory.
    shared_mem[shared_tid] = sum;
    __syncthreads(); // Wait for each thread to finish

    // Parallel reduction optimized for any block sizes
    for (int s = 1; s < blockDim.x; s *= 2) { // Parallel “folding” reduction, combining pairs of partial sums iteratively.
        int index = 2 * s * shared_tid; // For s=1, thread 0 processes shared_mem[0] and shared_mem[1], thread 1 processes shared_mem[2] and shared_mem[3], etc.
        if (index + s < blockDim.x) { // Prevents out-of-bounds memory access, making the reduction safe for non-power-of-2 block sizes
            shared_mem[index] += shared_mem[index + s]; // Combine the two partial sums
        }
        __syncthreads();
    }

    // Thread 0 writes the block's sum to global memory
    if (shared_tid == 0) {
        c[blockIdx.x] = shared_mem[0]; // This is the final sum for this block
    }
}

bool check(float cpuAnswer, float gpuAnswer, float tolerance) {
    double percentError = abs((gpuAnswer - cpuAnswer) / cpuAnswer) * 100.0;
    printf("\n\n percent error = %lf\n", percentError);
    
    return percentError < tolerance;
}

long elaspedTime(struct timeval start, struct timeval end) {
    long startTime = start.tv_sec * 1000000 + start.tv_usec;
    long endTime = end.tv_sec * 1000000 + end.tv_usec;
    return endTime - startTime;
}

void cleanUp() {
    cudaFreeHost(A_CPU); // To avoid segmentation faults
    cudaFreeHost(B_CPU); 
    cudaFreeHost(C_CPU);
    
    cudaFree(A_GPU); 
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU); 
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU);
    cudaErrorCheck(__FILE__, __LINE__);
}

int main(int argc, char *argv[]) {
    timeval start, end;
    long timeCPU, timeGPU;
    
    // Check for command-line argument
    if (argc != 2) {
        printf("Usage: %s <vector_length>\n", argv[0]);
        exit(1);
    }

    // Parse vector length from command-line argument
    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Error: Vector length must be a positive integer\n");
        exit(1);
    }

    // Set up devices (needs N for GridSize calculation)
    setUpDevices(N);
    
    // Allocate memory
    allocateMemory(N);
    
    // Initialize vectors
    innitialize(N);

    // CPU dot product
    gettimeofday(&start, NULL);
    dotProductCPU(A_CPU, B_CPU, N);
    DotCPU = C_CPU[0];
    gettimeofday(&end, NULL);
    timeCPU = elaspedTime(start, end);

    // GPU dot product
    gettimeofday(&start, NULL);
    
    cudaMemcpyAsync(A_GPU, A_CPU, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(B_GPU, B_CPU, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    
    cudaMemset(C_GPU, 0, GridSize.x * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventRecord(kernelStart);

	// Kernel launch with dynamic shared memory allocation
    dotProductGPU<<<GridSize, BlockSize, BlockSize.x * sizeof(float)>>>(A_GPU, B_GPU, C_GPU, N);
    cudaErrorCheck(__FILE__, __LINE__);
    
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    float kernelTimeMs;
    cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop);
    
    cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
    
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);
    
    DotGPU = 0.0f;
    for (int i = 0; i < GridSize.x; i++)
        DotGPU += C_CPU[i];
    
    gettimeofday(&end, NULL);
    timeGPU = elaspedTime(start, end);
    
    if (!check(DotCPU, DotGPU, Tolerance)) {
        printf("\n\n Something went wrong in the GPU dot product.\n");
        printf(" CPU Result = %f\n", DotCPU);
        printf(" GPU Result = %f\n", DotGPU);
        printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
        printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
    } else {
        printf("\n\n You did a dot product correctly on the GPU");
        printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
        printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
        printf("\n The CPU result was %f", DotCPU);
        printf("\n The GPU result was %f", DotGPU);
    }
	
    cleanUp();
    printf("\n\n");
    return 0;
}
