/*
 * util.h
 *
 *  Created on: 20/09/2014
 *      Author: josericardo
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define EPISILON 0.00001f

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
	    system("PAUSE>null");												\
		exit(1);															\
	} }

struct StopWatch{
	clock_t _start;
	clock_t _end;

	void start(){
		_start = clock();
	}

	void stop(){
		_end = clock();
	}

	float elapsedTime(){
		return (float)(((_end - _start) / (float)CLOCKS_PER_SEC ) * 1000.0f);
	}
};


/*matrix multiplication CPU*/
void MatrixMulHost(float *Md , float *Nd , float *Pd , int width)
{
	float sum = 0;

	for ( int row = 0 ; row < width; row++ )
	{
		for ( int col = 0 ; col < width; col++ )
	    {
			for ( int k = 0 ; k < width; k++ )
	        {
				sum += Md[row * width + k ] * Nd[ k * width + col];
	        }

			Pd[row*width + col] = sum;
			sum = 0;
	    }
	}
}

/* Test matrix equality */
bool TestMatrixResult(float *Md, float *Nd, const int width)
{
	for (int row = 0; row < width; row++){
		for (int col = 0; col < width; col++){
			if (fabs(Md[row * width + col]) - fabs(Nd[row * width + col]) > EPISILON)
				return false;
		}
	}

	return true;
}

/* Print a matrix */
void printMatrix(float *m, float w, float h){
   int i, j;

   printf("\n");

   for (j = 0; j < h; j++){
      for (i = 0; i < w; i++){
         int k = j * w + i;
         printf("%.2f ", m[k]);
      }
      printf("\n");
   }

}

void printMatrix(int *m, float w, float h){
   int i, j;

   printf("\n");

   for (j = 0; j < h; j++){
      for (i = 0; i < w; i++){
         int k = j * w + i;
         printf("%d ", m[k]);
      }
      printf("\n");
   }

}


#endif /* UTIL_H_ */
