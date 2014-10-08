#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/util.h"
#include "../common/App.h"

// Maze Values
//  0 -> Free position
//  1 -> Obstacle
//  2 -> Start Point
//  3 -> End Point
#define MAZE_FREE_POSITION 	0
#define MAZE_OBSTACLE 		1
#define MAZE_START_POINT	2
#define MAZE_END_POINT 		3

__device__  __host__ int getXY(int x, int y, int width)
{
	return y * width + x;
}

/*matrix multiplication kernels*/
__global__ void
MatrixMulDevice(int *A, int *B)
{
	unsigned int width  = gridDim.x * blockDim.x;
	unsigned int height = gridDim.y * blockDim.y;
	unsigned int x      = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y      = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset = x + y * width;

	if (offset < (width*width))
	{
		// C[getXY(x, y, width)] = 0;

		// float c = 0;

		// for (int i = 0; i < width; i++)
		// {
		// 	c += A[getXY(x, i, width)] * B[getXY(i, y, width)];
		// }

		// C[getXY(x, y, width)] = c;
	}
}


void generateMaze(int *maze, int width, int height)
{
	int index = 0;
	int value = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			// Generate walls
			if (i == 0 || i == height-1 || j == 0 || j == width-1)
			{
				value = MAZE_OBSTACLE;
			}

			index = i * width + j;
			maze[index] = value;
		}
	}
}

// main routine
int main()
{
	const int WIDTH  = 32;
	const int HEIGHT = 32;
	const int SIZE 	 = WIDTH * HEIGHT;

	int *maze, *maze_result;

	 // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, 0));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    if (sizeof(void *) != 8)
    {
        fprintf(stderr, "Unified Memory requires compiling for a 64-bit system.\n");
		cudaDeviceReset();
#ifdef _WIN32
	USER_PAUSE;
#endif
		exit(EXIT_SUCCESS);
    }

    if (((deviceProp.major << 4) + deviceProp.minor) < 0x30)
    {
        fprintf(stderr, "requires Compute Capability of SM 3.0 or higher to run.\nexiting...\n");
        cudaDeviceReset();
#ifdef _WIN32
	USER_PAUSE;
#endif
        exit(EXIT_SUCCESS);
    }

	//Reset no device
	CUDA_CHECK_RETURN(cudaDeviceReset());

	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&maze, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&maze_result, SIZE * sizeof(int)));

	//Generate Maze
	generateMaze(maze, WIDTH, HEIGHT);

	// Copy Maze to Maze results matrix
	int index;
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			index = i * WIDTH + j;
			maze_result[index] = maze[index];
		}
	}

	printf("\nPathfinder - GPU\n");
	printf("Maze size: %d x %d - memory: [global]\n", WIDTH, HEIGHT);

#ifdef _WIN32
	USER_PAUSE;
#endif

	// Call Kernel
	dim3 threadsPerBlock(32, 32);
	dim3 grid(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);

	MatrixMulDevice << <grid, threadsPerBlock >> > (maze, maze_result);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// Showing results
	printf("\nPathfinder - GPU\n");
	printMatrix(maze_result, WIDTH, HEIGHT);
#ifdef _WIN32
	USER_PAUSE;
#endif

	// 5 - Free memory
	cudaFree(maze);
	cudaFree(maze_result);

	printf("End\n");
#ifdef _WIN32
	USER_PAUSE;
#endif

	return 0;
}

