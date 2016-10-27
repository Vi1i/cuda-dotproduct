/*
 * simple.cu
 * includes setup funtion called from "driver" program
 * also includes kernel function 'cu_fillArray()'
 */

#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#define BLOCK_SIZE 32

// The __global__ directive identifies this function as a kernel
// Note: all kernels must be declared with return type void 
__global__ void cu_fillArray (int *array_d)
{
    int x;

    // Note: CUDA contains several built-in variables
    // blockIdx.x returns the blockId in the x dimension
    // threadIdx.x returns the threadId in the x dimension
    x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    array_d[x] = x;
}


// This function is called from the host computer.
// It manages memory and calls the function that is executed on the GPU
extern "C" void fillArray (int *array, int arraySize)
{
    //a_d is the GPU counterpart of the array that exists in host memory 
    int *array_d;
    cudaError_t result;

    // allocate space in the device 
    result = cudaMalloc ((void**) &array_d, sizeof(int) * arraySize);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed.");
        exit(1);
    }

    //copy the array from host to array_d in the device 
    result = cudaMemcpy (array_d, array, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed.");
        exit(1);
    }

    // set execution configuration
    dim3 dimblock (BLOCK_SIZE);
    dim3 dimgrid (arraySize/BLOCK_SIZE);

    // actual computation: Call the kernel
    cu_fillArray <<<dimgrid, dimblock>>> (array_d);

    // transfer results back to host
    result = cudaMemcpy (array, array_d, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed.");
        exit(1);
    }

    // release the memory on the GPU 
    result = cudaFree (array_d);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFree failed.");
        exit(1);
    }
}

