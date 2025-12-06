// Name: Nathan S
// Optimizing nBody GPU code. 
// nsys profile -o nbody_report ./temp 8192 0 // To run NVIDIA Nsight Systems
// nvprof ./temp 8192 0 // Terminal profiler not supported for 8.0 and higher use nsys instead
// on RTX 40 series compute capability is 8.9 so sm_89
// on RTX 30 series compute capability is 8.6 so sm_86
// on RTX 20 series compute capability is 7.5 so sm_75
// on GTX 10 series compute capability is 6.1 so sm_61
// on GTX 900 series compute capability is 5.2 so sm_52

// nvcc -O3 --use_fast_math -maxrregcount=64 -arch=sm_75 V_nBodySpeedChallenge4.cu -o temp -lglut -lm -lGLU -lGL -Xptxas="-dlcm=ca,-O3" \
-Xcompiler "-O3 -march=native -ffast-math" -lX11 -lpthread -ldl -lrt

/*
 What to do:
 This is some lean n-body code that runs on the GPU for any number of bodies (within reason). Take this code and make it 
 run as fast as possible using any tricks you know or can find (Like using NVIDIA Nsight Systems). Keep the same general 
 format so we can time it and compare it with others' code. This will be a competition.
 
 First place: 20 extra points on this HW
 Second place: 15 extra points on this HW
 Third place: 10 extra points on this HW
 
 To focus more on new ideas rather than just using a bunch of if statements to avoid going out of bounds, N will be a power 
 of 2 and 256 < N < 262,144. Put a check in your code to make sure this is true. The code most run on any power of 2 bodies
 also the final picture most look close to the same as it did before the speedup or something went wrong in the code.

 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
 
 Use this code (before your changes) as the baseline code to check your nbody speedup.
*/

/*
 Purpose:
 To use what you have learned in this course to optimize code with the add of NVIDIA Nsight Systems.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define UNROLL _Pragma("unroll 8")

// Define Constant at Compile Time
constexpr int BLOCK_SIZE = 256; // 256 was fastest in testing.
constexpr float PI = 3.14159265359f;
constexpr int DRAW_RATE = 10;

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
constexpr float G = 10.0f;
constexpr float H = 10.0f;
constexpr float LJP = 2.0; 
constexpr float LJQ = 4.0; // Since these 2 are constant known values, already pre-computed below


// Globals
int N, DrawFlag;
float4 *P, *V, *F, *M; // Using float4 for better memory alignment but "w" component is unused
//float *M; 
float4 *PGPU, *VGPU, *FGPU, *MGPU; // MGPU never used, only for the vibe
//float *MGPU;
float GlobeRadius, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
__global__ void matMulWarmUp(const float*, const float*, float*, int);
void runWarmUp();
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void getForces(float4 *, float4 *, float, float, int);
__global__ void moveBodies(float4 *, float4 *, float4 *, float, float, int);
void nBody();
int main(int, char**);

// Performs C = A * B for GPU Warm-up
__global__ void matMulWarmUp(const float* A, const float* B, float* C, int N) {
    // Get the global index for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        // Simple matrix multiplication
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Warm-Up Execution Function
void runWarmUp() {
    // HOST or DEVICE SETUP
    int N = 65536;
    size_t size = N * N * sizeof(float);
    
    // Allocate Host and Device Memory
    float *h_A, *h_B;
    float *d_A, *d_B, *d_C;

    // Allocate host memory for initialization only
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);

    // Initialize with non-zero values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)i + 0.5f;
        h_B[i] = (float)i + 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // LAUNCH WARM-UP KERNEL
    int blockSize = 256;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    //printf("Starting GPU Warm-up with Matrix Size: %d x %d\n", N, N);
    
    // Launch the kernel
    matMulWarmUp<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    // SYNCHRONIZE AND CLEAN UP
    cudaDeviceSynchronize(); 
    printf("Warm-up complete. GPU clocks should now be ready for use.\n\n");

    // Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Clean up host memory
    free(h_A);
    free(h_B);
}

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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
	//cudaErrorCheck(__FILE__, __LINE__);
	
	//glColor3d(1.0,1.0,0.5);
	glColor3d(1, 1, 1); // This color is slightly faster, not really
	for(i = 0; i < N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	runWarmUp(); // Warm up the GPU

	drawPicture();
	gettimeofday(&start, NULL);
    nBody();
    cudaDeviceSynchronize();
	//cudaErrorCheck(__FILE__, __LINE__);
    gettimeofday(&end, NULL);
    drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
	printf(" The compute time was %f seconds.\n\n", float(computeTime)/1000000.0f);
}

constexpr float computeDiameter(float H, float G, float LJP, float LJQ)
{
    //const float invG = 1.0f / G;
    //const float invExp = 1.0f / (LJQ - LJP);
	const float invExp = 0.5f; // LJQ = 4, LJP = 2, so invExp = 1/2 = 0.5
    //const float ratio = H * invG;
	const float ratio = 1.0f; // derived from H/G = 10/10 = 1

    //return expf(logf(ratio) * invExp);  // faster than powf or pow
	return 1.0f; // Because its doing the square root of 1 on this code
}  // This is the value where the force is zero for the L-J type force.


void setup()
{
    float randomAngle1, randomAngle2, randomRadius;
    float d, dx, dy, dz;
    int test;
    	
    BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1) / BlockSize.x + 1; // Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
	
    Damp = 0.5f;
    	
    M = (float4*)malloc(N*sizeof(float4));
    cudaMallocHost(&P, N * sizeof(float4));
    V = (float4*)malloc(N*sizeof(float4));
    F = (float4*)malloc(N*sizeof(float4));
    	
    cudaMalloc(&MGPU,N*sizeof(float4));
	//cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU,N*sizeof(float4));
	//cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU,N*sizeof(float4));
	//cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU,N*sizeof(float4));
	//cudaErrorCheck(__FILE__, __LINE__);
    	
	// Compute once before the loop
	//Diameter = computeDiameter(H, G, LJP, LJQ);
	//Diameter = 1.0f; // Pre-computed based on this code known values
	Radius = 0.5f;

	// Since the H, G, LJP, and LJQ values are constant in this code we can pre-compute the diameter and radius values.
	// constexpr float localDiameter = 1.0f;
	// constexpr float Radius = 0.5f;

	// float DiameterSquared = Diameter * Diameter;
	//constexpr float DiameterSquared = 1.0f; // Since Diameter = 1.0f

	constexpr float invRANDMAX = 1.0f / (float)RAND_MAX;
    constexpr float inv_packing_ratio = 1.47058823529f;  // 1/0.68
    constexpr float sphere_volume_constant = 0.238732414638f;  // 3/(4*PI)
    constexpr float twoPi = 6.28318530718f;
	//constexpr float fourOverThree = 1.33333333333f;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	//float totalVolume = float(N)*(fourOverThree)*PI*Radius*Radius*Radius;
	float totalVolume = float(N)*0.5235987755982988f; // Pre-computed (4/3)*PI*(0.5^3)
	totalVolume *= inv_packing_ratio;  // Instead of /= 0.68
    float totalRadius = cbrtf(sphere_volume_constant * totalVolume); // cbrtf = cube root which is faster than pow with 1/3
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the global sphere and setting the initial velosity, initial force, and mass.
	UNROLL for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			// Avoiding division
			randomAngle1 = ((float)rand() * invRANDMAX) * twoPi;
            randomAngle2 = ((float)rand() * invRANDMAX) * PI;
            randomRadius = ((float)rand() * invRANDMAX) * GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			P[i].w = 0.0f;  // padding
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = dx*dx + dy*dy + dz*dz; // No more sqrt
				if(d < 1.0f)
				{
					test = 0;
					break;
				}
			}
		}
	
		// Initial velocities
        V[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Initial forces
        F[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Mass stored in x, rest are padding with zeros
        M[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // Although honestly we would never used this
		// Only here for the vibe, chillin, killin
	}
	
	cudaMemcpy(PGPU, P, N*sizeof(float4), cudaMemcpyHostToDevice);
	//cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(VGPU, V, N*sizeof(float4), cudaMemcpyHostToDevice);
	//cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(FGPU, F, N*sizeof(float4), cudaMemcpyHostToDevice);
	//cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(MGPU, M, N*sizeof(float4), cudaMemcpyHostToDevice);
	//cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing go to the nBody window and type s.\n");
	printf("\n To quit type q in the nBody window.\n");
}

// *__restrict__ means "promise me you are the only one here so I don't have to check"
// no other independently declared pointer in the program will be used to access the same block of memory that "p" points to.
__global__ void getForces(float4 *__restrict__ p, float4 *__restrict__ f, float g, float h, int n) // velocity "v" and mass "m" are not needed
{
    __shared__ float4 sharedPosition[BLOCK_SIZE];
	//constexpr float myMass = 1.0f; // I just realized the mass is constant so is not needed

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float4 myPosition = make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 myForce = make_float4(0.0f,0.0f,0.0f,0.0f);
    
    // Load my position and mass, CONSTANT in this thread so no nesed to reload inside the loop
    if (i < n) { // ldcg uses L2 cache might be slower but good for large N, but ldg uses L1 which is faster, honestly I tried they're about the same speed
		myPosition.x = __ldg(&p[i].x);
		myPosition.y = __ldg(&p[i].y); // These 3 are loading the i-th x,y,z position
		myPosition.z = __ldg(&p[i].z);
		//myMass = __ldg(&m[i].x); // Load the i-th mass
	}
    
    // Pre-compute constant terms (done once per thread, not per interaction)
    //float g_mass = g;
    //float h_mass = h;
    
    // Process all particles in chunks / tiles of size BLOCK_SIZE
	UNROLL for (int tile = 0; tile < gridDim.x; tile++) { // Loop over tiles
        // Load tile of particles into shared memory
        int idx = tile * blockDim.x + threadIdx.x;

		// Same here, using "?" for single line if statement to avoid out-of-bounds access
        sharedPosition[threadIdx.x] = (idx < n) ? p[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f); // Reduce conditional divergence and initialize unused threads to zero
        __syncthreads(); // Make sure each thread receives the correct values before calculation
        
        // Compute forces with particles in this chunk
        if (i < n) {
            UNROLL for (int j = 0; j < blockDim.x; j++) {
                int globalJ = tile * blockDim.x + j;
                if (globalJ < n && i != globalJ) {
                    float dx = sharedPosition[j].x - myPosition.x;
                    float dy = sharedPosition[j].y - myPosition.y;
                    float dz = sharedPosition[j].z - myPosition.z;
                    float d2 = dx*dx + dy*dy + dz*dz; // Squared distance
                    
					// Based on Quake III algorithm for fast inverse square root
					// On modern NVIDIA GPUs, the CUDA compiler already provides a hardware-level intrinsic that does exactly what the Quake III function does
					// This is an optimized version of that and without using bit manipulation hacks.
                    float invd = rsqrtf(d2); // ≈ 1/sqrt(d2), d2 being the squared distance
                    float invd2 = invd * invd; // = 1/d^2
                    float invd4 = invd2 * invd2; // = 1/d^4
                    
                    // Use pre-computed mass terms
                    //float massTerm = sharedMass[j] * (g_mass*invd2 - h_mass*invd4);
					float massTerm = g*invd2 - h*invd4;
                    
					// Fused Multiply Add (fmaf) is faster than doing the multiply and add separately
					// fmaf is more portable C function while __fmaf_rz is rounded towards 0 and CUDA specific function
					// On my testing fmaf is more stable and faster than __fmaf_rz
                    myForce.x = fmaf(massTerm * dx, invd, myForce.x);
                    myForce.y = fmaf(massTerm * dy, invd, myForce.y);
                    myForce.z = fmaf(massTerm * dz, invd, myForce.z);
                }
            }
        }
        __syncthreads();
    }
    
    // Write result back to global memory
    if (i < n) {
        f[i] = myForce;
    }
}

__global__ void moveBodies(float4 *__restrict__ p, float4 *__restrict__ v, float4 *__restrict__ f, float dt, float t, int n) // mass is not needed
{	
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i < n)
	{
		constexpr float halfDt = 0.00005f;
		constexpr float damp = 0.5f;
		//float mass = m[i].x; // mass stored in x
		//float inv_mass; 1/1 is 1 LOL
		//float inv_mass = __frcp_rn(mass); // __frcp_rn for inverse Single Precision floating-point numbers
		//asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_mass) : "f"(mass)); // PTX assembly instruction

		if(t == 0.0f)
		{
			v[i].x += ((f[i].x - damp * v[i].x)) * halfDt;
			v[i].y += ((f[i].y - damp * v[i].y)) * halfDt;
			v[i].z += ((f[i].z - damp * v[i].z)) * halfDt;
		}
		else
		{
			v[i].x += ((f[i].x - damp * v[i].x))*dt;
			v[i].y += ((f[i].y - damp * v[i].y))*dt;
			v[i].z += ((f[i].z - damp * v[i].z))*dt;
		}

		p[i].x += v[i].x * dt;
		p[i].y += v[i].y * dt;
		p[i].z += v[i].z * dt;
	}
}

void nBody() {
    cudaStream_t computeStream, transferStream;
    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&transferStream);

    int drawCount = 0;
    float t = 0.0f;
    constexpr float dt = 0.0001f;
	constexpr float run_time = 1.0;

    while (t < run_time) {
        // Launch force calculation and body movement in compute stream
        getForces<<<GridSize, BlockSize, 0, computeStream>>>(PGPU, FGPU, G, H, N);
        //cudaErrorCheck(__FILE__, __LINE__);
        moveBodies<<<GridSize, BlockSize, 0, computeStream>>>(PGPU, VGPU, FGPU, dt, t, N);
        //cudaErrorCheck(__FILE__, __LINE__);

        // Perform data transfer for visualization in transfer stream
        if (drawCount == DRAW_RATE && DrawFlag) {
            cudaMemcpyAsync(P, PGPU, N * sizeof(float4), cudaMemcpyDeviceToHost, transferStream);
            cudaStreamSynchronize(transferStream);
            drawPicture();
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }

    // Clean up streams
    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(transferStream);
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}

	N = atoi(argv[1]);
	DrawFlag = atoi(argv[2]);

	// Validate N
    if (N <= 256)
    {
        printf("\nError: Number of bodies (%d) must be greater than 256.\n", N);
        exit(1);
    }
    if (N >= 262144)
    {
        printf("\nError: Number of bodies (%d) must be less than 262144.\n", N);
        exit(1);
    }
    if ((N & (N - 1)) != 0) // Bitwise check for power of 2
    {
        printf("\nError: Number of bodies (%d) must be a power of 2 (e.g., 512, 1024, 2048, ..., 131072).\n", N);
        exit(1);
    }
	setup();
	
	constexpr int XWindowSize = 1000;
	constexpr int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Modified nBody Challenge");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	cudaFreeHost(P); // Free pinned memory
    free(V);
    free(F);
    free(M);
    cudaFree(MGPU);
    cudaFree(PGPU);
    cudaFree(VGPU);
    cudaFree(FGPU);
	return 0;
}









