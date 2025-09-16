// Name: Nathanael Solagratia
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <math.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
// I changed it to 768x768 to show it works beside 1024x1024
unsigned int WindowWidth = 768;
unsigned int WindowHeight = 768;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float, float, float, float, float);
void display(void);

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

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int width, int height)
{
	// Ensure we do not go out of bounds and work with any window size
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

	// Checking if the thread is within the image bounds
    if (idx >= width || idy >= height) return;

	// Assigning each thread its x and y value of its pixel.
    float x = xMin + dx * idx;
    float y = yMin + dy * idy;

    int count = 0;
    float mag = sqrtf(x*x + y*y);
    float tempX;
    while (mag < MAXMAG && count < MAXITERATIONS) {
        tempX = x;
        x = x*x - y*y + A;
        y = 2.0f * tempX * y + B;
        mag = sqrtf(x*x + y*y);
        count++;
    }
    
    int id = 3 * (idy * width + idx);
	if (count == MAXITERATIONS) {
		// Inside - Deep black
		pixels[id] = 0.0f;
		pixels[id + 1] = 0.0f;
		pixels[id + 2] = 0.0f;
	} else {
		// Outside - Bright rainbow bands
		// Based on Sinusoidal Rainbow
		float mult = 0.3f;
		// mult => 0.3f for blue starting point, mult => 0.6f for red starting point, mult => 0.9f for green starting point
		float bands = fmodf((float)count * mult, 1.0f); //
		float phase = bands *  6.28318f;  // 2π for full cycle
		pixels[id] = 0.3f + 0.7f * sinf(phase); // Red = radian value of 0 or 360 degrees
		pixels[id + 1] = 0.3f + 0.7f * sinf(phase + 2.094f); // Green = radian value of +120 degrees
		pixels[id + 2] = 0.3f + 0.7f * sinf(phase + 4.188f); // Blue = radian value of +240 degrees
		mult += 0.05f; // Makes it more chaotic as the colors change faster
	}
	// bands acts as a normalized position [0, 1] within a color cycle by taking the fmodulus of count*multiplier
	// phase converts the normalized bands value into a radian value for the sine function
	// sinf(phase) ranges from [-1, +1]
	// 0.3f for minimum brightness (prevents pure black)
	// 0.7f for the amplitude (color intensity range)
	// Higher multiplier = more color bands, lower multiplier = less color bands
}


void display(void) 
{ 
    float *pixelsCPU, *pixelsGPU; 
    float stepSizeX, stepSizeY;

    // Allocate CPU and GPU memory
    pixelsCPU = (float *)malloc(WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaMalloc(&pixelsGPU, WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    
    stepSizeX = (XMax - XMin) / ((float)WindowWidth);
    stepSizeY = (YMax - YMin) / ((float)WindowHeight);
    
    // Use 2D blocks and grids
    dim3 blockSize(16, 16);  // 256 threads per block
	// Compute grid size to cover the whole window
    dim3 gridSize((WindowWidth  + blockSize.x - 1) / blockSize.x,
                  (WindowHeight + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY,
                                         WindowWidth, WindowHeight);
    cudaErrorCheck(__FILE__, __LINE__);
    
    // Copy result back
    cudaMemcpy(pixelsCPU, pixelsGPU, WindowWidth * WindowHeight * 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
    
    // Draw pixels
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
    glFlush(); 
    
    // Free GPU memory
    cudaFree(pixelsGPU);
    free(pixelsCPU);
}


int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	// Set window position before creating window
	int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
	int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
	// At the center of the screen, calculate the median values
	glutInitWindowPosition(
		(screenWidth - WindowWidth) / 2,
		(screenHeight - WindowHeight) / 2
	);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Lysergic acid diethylamide Fractal");
   	glutDisplayFunc(display);
   	glutMainLoop();
}



