/*
 * blockAndThread.cu
 * includes setup funtion called from "driver" program
 * also includes kernel function 'cu_fillArray()'
 */

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

// The __global__ directive identifies this function as a kernel
// Note: all kernels must be declared with return type void 
__global__ void cu_fillArray (int *block_d, int *thread_d)
{
    int x;

    // Note: CUDA contains several built-in variables
    // blockIdx.x returns the blockId in the x dimension
    // threadIdx.x returns the threadId in the x dimension
    x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    block_d[x] = blockIdx.x;
    thread_d[x] = threadIdx.x;
}


// This function is called from the host computer.
// It manages memory and calls the function that is executed on the GPU
extern "C" void fillArray (int *block, int *thread, int arraySize)
{
    // block_d and thread_d are the GPU counterparts of the arrays that exists in host memory 
    int *block_d;
    int *thread_d;
    cudaError_t result;

    // allocate space in the device 
    result = cudaMalloc ((void**) &block_d, sizeof(int) * arraySize);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (block) failed.");
        exit(1);
    }
    result = cudaMalloc ((void**) &thread_d, sizeof(int) * arraySize);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (thread) failed.");
        exit(1);
    }

    //copy the arrays from host to the device 
    result = cudaMemcpy (block_d, block, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy host->dev (block) failed.");
        exit(1);
    }
    result = cudaMemcpy (thread_d, thread, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy host->dev (thread) failed.");
        exit(1);
    }

    // set execution configuration
    dim3 dimblock (BLOCK_SIZE);
    dim3 dimgrid (arraySize/BLOCK_SIZE);

    // actual computation: Call the kernel
    cu_fillArray <<<dimgrid, dimblock>>> (block_d, thread_d);

    // transfer results back to host
    result = cudaMemcpy (block, block_d, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy host <- dev (block) failed.");
        exit(1);
    }
    result = cudaMemcpy (thread, thread_d, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy host <- dev (thread) failed.");
        exit(1);
    }

    // release the memory on the GPU 
    result = cudaFree (block_d);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFree (block) failed.");
        exit(1);
    }
    result = cudaFree (thread_d);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFree (thread) failed.");
        exit(1);
    }
}

