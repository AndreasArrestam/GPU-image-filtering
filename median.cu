#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10
#define BLOCKSIZE 32
#define kernelSizeX 9
#define kernelSizeY 9

__device__ unsigned char median_from_histogram(int* hist, int kernelSize) {
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += hist[i];
        if (sum >= kernelSize / 2)
            return (unsigned char)i;
    }
    return 0;
}
__global__ void median_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey)
{ 
 
	int globalThreadIndexx = blockIdx.x*blockDim.x + threadIdx.x;
	int globalThreadIndexy = blockIdx.y*blockDim.y + threadIdx.y;

  	int dy, dx;
	unsigned int sumR, sumG, sumB;
	int sharedW = BLOCKSIZE + 2*kernelSizeX;
	int sharedH = BLOCKSIZE + 2*kernelSizeY;

	int shared_x = threadIdx.x + kernelSizeX;
	int shared_y = threadIdx.y + kernelSizeY;

	
	__shared__	unsigned char sharedMemR[(BLOCKSIZE + 2*kernelSizeX) * (BLOCKSIZE + 2*kernelSizeY)];
	__shared__	unsigned char sharedMemG[(BLOCKSIZE + 2*kernelSizeX) * (BLOCKSIZE + 2*kernelSizeY)];
	__shared__	unsigned char sharedMemB[(BLOCKSIZE + 2*kernelSizeX) * (BLOCKSIZE + 2*kernelSizeY)];
	int divby = (2*kernelSizeX+1)*(2*kernelSizeY+1); // This should stay or be computed only once

	//populate shared memory
	for (int shared_y = threadIdx.y; shared_y < sharedH; shared_y += BLOCKSIZE)
	{
		for (int shared_x = threadIdx.x; shared_x < sharedW; shared_x += BLOCKSIZE)
			{
				// compute the global coordinates corresponding to (shared_x, shared_y)
				int gx = blockIdx.x * BLOCKSIZE + shared_x - kernelSizeX;
				int gy = blockIdx.y * BLOCKSIZE + shared_y - kernelSizeY;

				// at image borders this will repeat the edge pixels otherwise normal
				gx = min(max(gx, 0), imagesizex - 1);
				gy = min(max(gy, 0), imagesizey - 1);

				int g_idx = gy * imagesizex + gx;
				int sidx = shared_y * sharedW + shared_x;

				sharedMemR[sidx] = image[g_idx*3 + 0];
				sharedMemG[sidx] = image[g_idx*3 + 1];
				sharedMemB[sidx] = image[g_idx*3 + 2];
			}
	}

	__syncthreads();

	if (globalThreadIndexx < imagesizex && globalThreadIndexy < imagesizey) // If inside image
    {
        int histR[256] = {0}, histG[256] = {0}, histB[256] = {0};

        // Compute median over 2D neighborhood
        for (int dy = -kernelSizeY; dy <= kernelSizeY; dy++) {
            for (int dx = -kernelSizeX; dx <= kernelSizeX; dx++) {
                int sx = threadIdx.x + dx + kernelSizeX;
                int sy = threadIdx.y + dy + kernelSizeY;
                int s_idx = sy * sharedW + sx;

                histR[sharedMemR[s_idx]]++;
                histG[sharedMemG[s_idx]]++;
                histB[sharedMemB[s_idx]]++;
            }
        }

        int kernelSize = (2*kernelSizeX+1) * (2*kernelSizeY+1);
        int out_idx = (globalThreadIndexy * imagesizex + globalThreadIndexx) * 3;
        out[out_idx + 0] = median_from_histogram(histR, kernelSize);
        out[out_idx + 1] = median_from_histogram(histG, kernelSize);
        out[out_idx + 2] = median_from_histogram(histB, kernelSize);
    }
	__syncthreads();
}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	float executionTime = 0.0;

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	
	//dim3 dimBlock()
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid(imagesizex/32,imagesizey/32);

	cudaEventRecord(start, 0);
	// Launch the kernel
	median_filter<<<dimGrid,dimBlock>>>(dev_input, dev_bitmap, imagesizex, imagesizey); 
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&executionTime, start, stop);
	printf("\nExecution time: %f\n", executionTime);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"baboon1.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Median filtering");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(kernelSizeX, kernelSizeY);

	glutMainLoop();
	return 0;
}
