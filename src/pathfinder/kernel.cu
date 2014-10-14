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
#define MAZE_SIZE_X 32
#define MAZE_SIZE_Y 32
#define MAZE_SIZE (MAZE_SIZE_X*MAZE_SIZE_Y)

#define MAZE_INVALID_POSITION -2
#define MAZE_UNKNOWN_POSITION -1

#define NUMBERS_MAX_COUNT 1024
#define FILE_BUFFER_READ_MAX_SIZE 1024

#define FILENAME "../data/maze.txt"

__device__  __host__ int getXY(int x, int y, int width)
{
	return y * width + x;
}

int loadNumbersFromFile(const char *filename, int *numberList)
{
    FILE *fd = NULL;

    fd = fopen(filename, "r");

    if (!fd) {
        printf("Error when trying to open file! Please, create maze.txt file on data folder!\n");

        return 0;
    }

    int itemIndex = 0;

    while (itemIndex < (MAZE_SIZE_X*MAZE_SIZE_Y) && fscanf(fd, "%ld", &numberList[itemIndex])) {
        ++itemIndex;
    }

    fclose(fd);

    return itemIndex;
}

__global__ void verify_last_state(int *maze,
								  int* maze_result,
								  int start_position,
								  int x, int y,
								  unsigned int width,
								  unsigned int height)
{
		int nx      		= threadIdx.x + x - 1;
		int ny      		= threadIdx.y + y - 1;
		int offset 			= nx + ny * width;
		int current_offset 	= x  + y * width;

		int current_value = 0;
		bool set_value = false;

		if (offset >= 0 && offset < (width*height))
		{
			current_value = maze[offset];
			set_value = current_value  >= 0 || offset == start_position;
		}

			//__syncthreads();

		if (set_value)
		{
			maze_result[current_offset] = offset;
		}
}

__global__ void get_path(int *maze, int *result, int start_position, int goal_position)
{
	unsigned int width  = gridDim.x * blockDim.x;
	unsigned int height = gridDim.y * blockDim.y;
	unsigned int x      = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y      = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset = x + y * width;

	if (offset < (width*height))
	{
		dim3 threadsPerNeighbor(3,3);

		bool found = false;
		int max_steps = MAZE_SIZE;

		while(!found && max_steps > 0)
		{
			__syncthreads();

			// get next states
			if (result[offset] == MAZE_UNKNOWN_POSITION)
			{
				verify_last_state<<<1, threadsPerNeighbor>>>(maze, result, start_position, x, y, width, height);
			}

			__syncthreads();
			cudaDeviceSynchronize();
			__syncthreads();

			maze[offset] = result[offset];

			__syncthreads();

			// is the goal point any of those states?
			found = result[goal_position] != MAZE_UNKNOWN_POSITION;

			max_steps--;

			__syncthreads();
		}
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
			else
			{
				value = MAZE_FREE_POSITION;
			}

			index = i * width + j;
			maze[index] = value;
		}
	}
}

void printGPUProperties()
{
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
		exit(EXIT_SUCCESS);
	}

	if (((deviceProp.major << 4) + deviceProp.minor) < 0x30)
	{
		fprintf(stderr, "requires Compute Capability of SM 3.0 or higher to run.\nexiting...\n");
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

// main routine
int main()
{
	int WIDTH  = MAZE_SIZE_X;
	int HEIGHT = MAZE_SIZE_Y;

	printGPUProperties();

	//Reset no device
	CUDA_CHECK_RETURN(cudaDeviceReset());

	int *maze, *maze_result;

	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&maze, MAZE_SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&maze_result, MAZE_SIZE * sizeof(int)));


	// Load Maze
	if (loadNumbersFromFile(FILENAME, maze) == 0)
	{
		// Generate Maze if can't load from file
		generateMaze(maze, WIDTH, HEIGHT);
	}

	// Initialize Maze results matrix
	int index;
	int START_POINT = 0, END_POINT = 0;

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			index = i * WIDTH + j;

			if (maze[index] == MAZE_OBSTACLE)
			{
				maze_result[index] = MAZE_INVALID_POSITION;
			}
			else
			{
				maze_result[index] = MAZE_UNKNOWN_POSITION;
			}

			if (maze[index] == MAZE_START_POINT)
			{
				START_POINT = index;
			}
			else if (maze[index] == MAZE_END_POINT)
			{
				END_POINT = index;
			}

			maze[index] = maze_result[index];
		}
	}

	printf("\nPathfinder - GPU\n");
	printf("Maze size: %d x %d - memory: [global]\n", WIDTH, HEIGHT);
	printf("Maze start at (%d) and ends at (%d)\n", START_POINT, END_POINT);

	dim3 threadsPerBlock(32,32);
	dim3 grid(MAZE_SIZE_X / threadsPerBlock.x, MAZE_SIZE_Y / threadsPerBlock.y);

	get_path<<<grid, threadsPerBlock>>>(maze, maze_result, START_POINT, END_POINT);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	int next_pos = 0;
	int step = 0;

	for (int i = 0; i < MAZE_SIZE; i++)
	{
		maze[i] = 0;
	}

	for (int pos = END_POINT; pos != START_POINT; pos = next_pos)
	{
		next_pos = maze_result[pos];
		maze[pos] = step;
		step++;
	}

	maze[START_POINT] = step;

	// Showing results
	printf("\nPathfinder - GPU\n");
	printMatrix(maze, WIDTH, HEIGHT);

	// 5 - Free memory
	cudaFree(maze);
	cudaFree(maze_result);

	CUDA_CHECK_RETURN(cudaDeviceReset());

	printf("End\n");

	return 0;
}

