// Name: Nathanael Solagratia
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <cuda_runtime.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824    //Real part of C
#define B  -0.1711   //Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float x, float y);

// CUDA error check function
void cudaErrorCheck(const char *file, int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
        exit(0);
    }
}

// CUDA kernel to compute the escape color for each pixel
__global__ void fractalKernel(float *pixels, float XMin, float XMax, float YMin, float YMax, int width, int height)
{
	// Each thread computes one pixel and avoid overlap by calculating its unique index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height)
    {
        float stepSizeX = (XMax - XMin) / width;
        float stepSizeY = (YMax - YMin) / height;
        
        float x = XMin + idx * stepSizeX;
        float y = YMin + idy * stepSizeY;
        
        float color = escapeOrNotColor(x, y); 
        
        int k = (idy * width + idx) * 3; // Red, Green, Blue for this pixel
        
        pixels[k] = color;    // Red channel (0 = black, 1 = red)
        pixels[k+1] = 0.0;    // Green channel
        pixels[k+2] = 0.0;    // Blue channel
    }
}

// Device function to check if the point escapes
__device__ float escapeOrNotColor(float x, float y)
{
    float mag, tempX;
    int count = 0;
    float maxMag = MAXMAG;
    int maxCount = MAXITERATIONS;

    mag = sqrt(x * x + y * y);
    
    while (mag < maxMag && count < maxCount)
    {
        tempX = x;
        x = x * x - y * y + A;
        y = (2.0 * tempX * y) + B;
        mag = sqrt(x * x + y * y);
        count++;
    }

    return (count < maxCount) ? 0.0 : 1.0;
}

void display(void) 
{
    float *d_pixels, *h_pixels;
    
    // Allocate memory on host (CPU)
    h_pixels = (float *)malloc(WindowWidth * WindowHeight * 3 * sizeof(float));
    
    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_pixels, WindowWidth * WindowHeight * 3 * sizeof(float));

    // Launch CUDA kernel
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((WindowWidth + blockSize.x - 1) / blockSize.x, (WindowHeight + blockSize.y - 1) / blockSize.y);
	// Match the number of pixels need to compute and each thread will be responsible for processing a single pixel
    fractalKernel<<<gridSize, blockSize>>>(d_pixels, XMin, XMax, YMin, YMax, WindowWidth, WindowHeight);
    cudaErrorCheck(__FILE__, __LINE__);
    
    // Copy the result back to host
    cudaMemcpy(h_pixels, d_pixels, WindowWidth * WindowHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Putting pixels on the screen
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, h_pixels);
    glFlush();

    // Free device memory
    cudaFree(d_pixels);
    free(h_pixels);
}

int main(int argc, char** argv) 
{ 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(WindowWidth, WindowHeight);
    glutCreateWindow("Fractals--Man--Fractals");
    glutDisplayFunc(display);
    glutMainLoop();
}


