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

#define NUMBERS_MAX_COUNT 1024
#define FILE_BUFFER_READ_MAX_SIZE 1024

#define FILENAME "../data/maze.txt"

#define GET_TRANSITION_MATRIX(Name, config) \
__global__ void \
Name(int *A, int *transitionMatrix) \
{\
	unsigned int width  = gridDim.x * blockDim.x;\
	unsigned int height = gridDim.y * blockDim.y;\
	unsigned int x      = blockIdx.x * blockDim.x + threadIdx.x;\
	unsigned int y      = blockIdx.y * blockDim.y + threadIdx.y;\
	unsigned int offset = x + y * width;\
\
	if (offset < (width*height))	{ \
		int nx = x;\
		int ny = y;\
		\
		config;\
		\
		int noffset = nx + ny * width;\
\
		if (noffset >= 0 && noffset < (width*height))\
		{\
			transitionMatrix[offset] = A[noffset] == MAZE_OBSTACLE? -1 : noffset;\
		}\
	}\
}


GET_TRANSITION_MATRIX(Up, nx=x;ny=y+1;);
GET_TRANSITION_MATRIX(Down, nx=x;ny=y-1;);
GET_TRANSITION_MATRIX(Left, nx=x-1;ny=y;);
GET_TRANSITION_MATRIX(Right, nx=x+1;ny=y;);
GET_TRANSITION_MATRIX(UpRight, nx=x+1;ny=y+1;);
GET_TRANSITION_MATRIX(UpLeft, nx=x-1;ny=y+1;);
GET_TRANSITION_MATRIX(DownRight, nx=x+1;ny=y-1;);
GET_TRANSITION_MATRIX(DownLeft, nx=x-1;ny=y-1;);

struct PathfindingData
{
	int *transitionUp;
	int *transitionDown;
	int *transitionLeft;
	int *transitionRight;
	int *transitionUpRight;
	int *transitionUpLeft;
	int *transitionDownRight;
	int *transitionDownLeft;
};

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

    while (itemIndex < (32*32) && fscanf(fd, "%ld", &numberList[itemIndex])) {
        ++itemIndex;
    }

    fclose(fd);

    return itemIndex;
}

/**
 * Pathfinding using an iterative approach and Manhattan distance as heuristic value
 **/
__global__ void
GetPathUsingManhattanDistance(int *A, int *B, int startPointX, int startPointY, int goalPointX, int goalPointY)
{
	unsigned int width  = gridDim.x * blockDim.x;
	unsigned int height = gridDim.y * blockDim.y;
	unsigned int x      = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y      = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset = x + y * width;

	if (offset < (width*height))
	{
		int currentValue = A[offset];

		if (currentValue != MAZE_OBSTACLE)
		{
			currentValue = currentValue == MAZE_FREE_POSITION? 0 : 1;

			int mx, my;

			mx = __sad(x,startPointX,0);
			my = __sad(y,startPointY,0);
			int manhattan_from_start = max(mx,my);

			mx = __sad(x,goalPointX,0);
			my = __sad(y,goalPointY,0);
			int manhattan_from_goal = max(mx,my);

			//B[offset] = currentValue * (manhattan);
			B[offset] = manhattan_from_start + manhattan_from_goal;

			__syncthreads();

			A[offset] = B[offset];

			__syncthreads();

			int path_value = A[getXY(startPointX, startPointY, width)];

			B[offset] = A[offset] == path_value? manhattan_from_start : 0;
		}
		else
		{
			B[offset] = width*height;
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

// main routine
int main()
{
	int WIDTH  = 32;
	int HEIGHT = 32;
	int SIZE 	 = WIDTH * HEIGHT;

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

	int *maze, *maze_result;
	PathfindingData* pathfindingData;

	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&maze, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&maze_result, SIZE * sizeof(int)));

	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData, sizeof(PathfindingData)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionUp, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionLeft, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionRight, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionDown, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionUpLeft, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionUpRight, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionDownLeft, SIZE * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&pathfindingData->transitionDownRight, SIZE * sizeof(int)));


	//Generate Maze
	//generateMaze(maze, WIDTH, HEIGHT);
	loadNumbersFromFile(FILENAME, maze);

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

	// Set start point
	int START_POINT_X = 3,
		START_POINT_Y = 4;
	int START_POINT = getXY(START_POINT_X, START_POINT_Y, WIDTH);
	maze[START_POINT] = MAZE_START_POINT;
	// Set end point
	int END_POINT_X = 10,
		END_POINT_Y = 18;
	int END_POINT = getXY(END_POINT_X, END_POINT_Y, WIDTH);
	maze[END_POINT] = MAZE_END_POINT;

	printf("\nPathfinder - GPU\n");
	printf("Maze size: %d x %d - memory: [global]\n", WIDTH, HEIGHT);

#ifdef _WIN32
	USER_PAUSE;
#endif

	dim3 threadsPerBlock(32, 32);
	dim3 grid(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);

	Up<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionUp);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	Down<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionDown);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	Right<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionRight);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	Left<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionLeft);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	UpRight<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionUpRight);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	UpLeft<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionUpLeft);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	DownRight<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionDownRight);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	DownLeft<<<grid, threadsPerBlock>>>(maze, pathfindingData->transitionDownLeft);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	/** Dummy approach - Get path using Manhattan Distance **/
	//
	//GetPathUsingManhattanDistance << <grid, threadsPerBlock >> > (maze, maze_result, START_POINT_X, START_POINT_Y, END_POINT_X, END_POINT_Y);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// Showing results
	printf("\nPathfinder - GPU\n");
	printMatrix(maze, WIDTH, HEIGHT);
#ifdef _WIN32
	USER_PAUSE;
#endif

	// 5 - Free memory
	cudaFree(maze);
	cudaFree(maze_result);
	cudaFree(pathfindingData->transitionDown);
	cudaFree(pathfindingData->transitionDownLeft);
	cudaFree(pathfindingData->transitionDownRight);
	cudaFree(pathfindingData->transitionLeft);
	cudaFree(pathfindingData->transitionRight);
	cudaFree(pathfindingData->transitionUp);
	cudaFree(pathfindingData->transitionUpLeft);
	cudaFree(pathfindingData->transitionUpRight);
	cudaFree(&pathfindingData);

	printf("End\n");
#ifdef _WIN32
	USER_PAUSE;
#endif

	return 0;
}

