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
#include <numeric>
#include <vector>

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10
#define BLOCKSIZE 32
#define kernelsize 9
// #define kernelSizeY 7



__global__ void horizontal_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey, const int* gaussianStencil)
{ 
  
	int globalThreadIndexx = blockIdx.x*blockDim.x + threadIdx.x;
	int globalThreadIndexy = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned int sumR, sumG, sumB;
	
	int sharedW = BLOCKSIZE + 2*kernelsizex;
	int sharedH = BLOCKSIZE + 2*kernelsizey;

	int shared_x = threadIdx.x + kernelsizex;
	int shared_y = threadIdx.y + kernelsizey;


	__shared__	unsigned char sharedMemR[(BLOCKSIZE + 2*kernelsize) * (BLOCKSIZE + 2*kernelsize)];
	__shared__	unsigned char sharedMemG[(BLOCKSIZE + 2*kernelsize) * (BLOCKSIZE + 2*kernelsize)];
	__shared__	unsigned char sharedMemB[(BLOCKSIZE + 2*kernelsize) * (BLOCKSIZE + 2*kernelsize)];
	
    int divby = 0;
    for(int i = 0; i < kernelsize; i++) {
        divby += gaussianStencil[i]; 
        
    }


	for (int shared_y = threadIdx.y; shared_y < sharedH; shared_y += BLOCKSIZE)
	{
		for (int shared_x = threadIdx.x; shared_x < sharedW; shared_x += BLOCKSIZE)
			{
				// compute the global coordinates corresponding to (shared_x, shared_y)
				int gx = blockIdx.x * BLOCKSIZE + shared_x - kernelsizex;
				int gy = blockIdx.y * BLOCKSIZE + shared_y - kernelsizey;

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
		// Filter kernel (simple box filter)
		sumR=0;sumG=0;sumB=0;
		for(int dx = -kernelsizex/2; dx <= kernelsizex/2; dx++) {
            int xx = shared_x + dx;
            sumR += sharedMemR[shared_y*sharedW + xx] * gaussianStencil[dx + kernelsizex/2];
            sumG += sharedMemG[shared_y*sharedW + xx] * gaussianStencil[dx + kernelsizex/2];
            sumB += sharedMemB[shared_y*sharedW + xx] * gaussianStencil[dx + kernelsizex/2];
        }
			__syncthreads();

		out[(globalThreadIndexy*imagesizex+globalThreadIndexx)*3+0] = sumR/divby;
		out[(globalThreadIndexy*imagesizex+globalThreadIndexx)*3+1] = sumG/divby;
		out[(globalThreadIndexy*imagesizex+globalThreadIndexx)*3+2] = sumB/divby;
	}
	__syncthreads();
}


__global__ void vertical_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey, const int* gaussianStencil)
{ 
  
	int globalThreadIndexx = blockIdx.x*blockDim.x + threadIdx.x;
	int globalThreadIndexy = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned int sumR, sumG, sumB;
	int sharedW = BLOCKSIZE + 2*kernelsizex;
	int sharedH = BLOCKSIZE + 2*kernelsizey;

	int shared_x = threadIdx.x + kernelsizex;
	int shared_y = threadIdx.y + kernelsizey;


	__shared__	unsigned char sharedMemR[(BLOCKSIZE + 2*kernelsize) * (BLOCKSIZE + 2*kernelsize)];
	__shared__	unsigned char sharedMemG[(BLOCKSIZE + 2*kernelsize) * (BLOCKSIZE + 2*kernelsize)];
	__shared__	unsigned char sharedMemB[(BLOCKSIZE + 2*kernelsize) * (BLOCKSIZE + 2*kernelsize)];
	
    int divby = 0;
    for(int i = 0; i < kernelsize; i++) {
        divby += gaussianStencil[i];         
    }


	for (int shared_y = threadIdx.y; shared_y < sharedH; shared_y += BLOCKSIZE)
	{
		for (int shared_x = threadIdx.x; shared_x < sharedW; shared_x += BLOCKSIZE)
			{
				// compute the global coordinates corresponding to (shared_x, shared_y)
				int gx = blockIdx.x * BLOCKSIZE + shared_x - kernelsizex;
				int gy = blockIdx.y * BLOCKSIZE + shared_y - kernelsizey;

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
		// Filter kernel (simple box filter)
		sumR=0;sumG=0;sumB=0;
		for(int dy = -kernelsizey/2; dy <= kernelsizey/2; dy++) {
            int yy = shared_y + dy;
            sumR += sharedMemR[yy*sharedW + shared_x] * gaussianStencil[dy + kernelsizey/2];
            sumG += sharedMemG[yy*sharedW + shared_x] * gaussianStencil[dy + kernelsizey/2];
            sumB += sharedMemB[yy*sharedW + shared_x] * gaussianStencil[dy + kernelsizey/2];
        }
		__syncthreads();
		out[(globalThreadIndexy*imagesizex+globalThreadIndexx)*3+0] = sumR/divby;
		out[(globalThreadIndexy*imagesizex+globalThreadIndexx)*3+1] = sumG/divby;
		out[(globalThreadIndexy*imagesizex+globalThreadIndexx)*3+2] = sumB/divby;
	}
	__syncthreads();
}
// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void sampleGaussian(float* vals, int radius)
{
	float s = radius / 3.0;
	for (int i = -radius; i <= radius; ++i)
	{
		double x = (float)i;
		vals[i+radius] = exp(-x*x / (2*s*s)) / sqrt(2*s*s*M_PI);
	}
}

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

    const int h_gaussianStencil[9] = { 1, 8, 28, 56, 70, 56, 28, 8, 1 };

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
    
    int* dev_gaussianStencil;
    cudaMalloc((void**)&dev_gaussianStencil, sizeof(h_gaussianStencil));
    cudaMemcpy(dev_gaussianStencil, h_gaussianStencil, sizeof(h_gaussianStencil), cudaMemcpyHostToDevice);
	
	//dim3 dimBlock()
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid(imagesizex/32,imagesizey/32);

	cudaEventRecord(start, 0);
	// Launch the kernel


	vertical_filter<<<dimGrid,dimBlock>>>(dev_input, dev_bitmap, imagesizex, imagesizey, 0, kernelsizey, dev_gaussianStencil); 
	cudaDeviceSynchronize();
    
	horizontal_filter<<<dimGrid,dimBlock>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, 0, dev_gaussianStencil); 
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
	glutCreateWindow("Gaussian filtering");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(kernelsize, kernelsize);
	
	glutMainLoop();
	return 0;
}
