/*******************************************************************************
 *
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 512

#define GPU_ERR_CHK(ans) { gpu_assert((ans), __FILE__, __LINE__); }
static void gpu_assert(cudaError_t code, const char *file, int line,
        bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort) {
            exit(code);
        }
    }
}

__global__ void cu_init(unsigned long long seed, curandState_t * states_d,
        size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        curand_init(seed, idx, 0, &states_d[idx]);
    }
}

__global__ void cugen_curand_array(curandState_t * states_d, int * array_d,
        size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        int r = curand_uniform(&states_d[idx]) * 100;
        array_d[idx] = r;
    }
}

extern "C" void gen_curand_array(unsigned long long seed, int * array,
        size_t size) {
    int blocks = ceil(size / ((float) BLOCK_SIZE));
    dim3 dimgrid (blocks);
    dim3 dimblock (BLOCK_SIZE);
    curandState_t * states_d;
    int * array_d;

    GPU_ERR_CHK(cudaMalloc((void **) &states_d, size *
                sizeof(curandState_t)));
    cu_init<<<dimgrid, dimblock>>>(seed, states_d, size);

    GPU_ERR_CHK(cudaMalloc((void **) &array_d, size * sizeof(int)));
    cugen_curand_array<<<dimgrid, dimblock>>>(states_d, array_d, size);

    GPU_ERR_CHK(cudaMemcpy(array, array_d, size * sizeof(int),
                cudaMemcpyDeviceToHost));

    GPU_ERR_CHK(cudaFree(states_d));
    GPU_ERR_CHK(cudaFree(array_d));
}

template <unsigned int block_size>
__global__ void cu_dot(int * a_d, int * b_d, int * block_results_d,
        size_t size) {
    extern __shared__ int cache[];

    unsigned int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    cache[tid] = 0;

    if(idx < size) {
        cache[tid] = a_d[idx] * b_d[idx];
    }

    __syncthreads();
    if(block_size >= 512) {
        if(tid < 256) {
            cache[tid] += cache[tid + 256];
        }
        __syncthreads();
    }

    if(block_size >= 256) {
        if(tid < 128) {
            cache[tid] += cache[tid + 128];
        }
        __syncthreads();
    }

    if(block_size >= 128) {
        if(tid < 64) {
            cache[tid] += cache[tid + 64];
        }
        __syncthreads();
    }

    if(tid < 32) {
        if(block_size >= 64) {
            cache[tid] += cache[tid + 32];
        }
        __syncthreads();

        if(block_size >= 32) {
            cache[tid] += cache[tid + 16];
        }
        __syncthreads();

        if(block_size >= 16) {
            cache[tid] += cache[tid + 8];
        }
        __syncthreads();

        if(block_size >= 8) {
            cache[tid] += cache[tid + 4];
        }
        __syncthreads();

        if(block_size >= 4) {
            cache[tid] += cache[tid + 2];
        }
        __syncthreads();

        if(block_size >= 2) {
            cache[tid] += cache[tid + 1];
        }
    }

    __syncthreads();
    if(tid == 0) {
        block_results_d[blockIdx.x] = cache[0];
    }
}

extern "C" void dot_product(long long * result, int * a, int * b, size_t size) {
    int * a_d;
    int * b_d;
    int blocks = 1;
    if(size > BLOCK_SIZE) {
        blocks = ceil(((float)size) / BLOCK_SIZE);
    }
    int cache_size = BLOCK_SIZE * sizeof(int); 
    int * block_results_d;
    int * block_results = (int *) malloc(blocks * sizeof(int));
    dim3 dimgrid (blocks);
    dim3 dimblock (BLOCK_SIZE);

    GPU_ERR_CHK(cudaMalloc((void **) &a_d, sizeof(int) * size));
    GPU_ERR_CHK(cudaMalloc((void **) &b_d, sizeof(int) * size));
    GPU_ERR_CHK(cudaMalloc((void **) &block_results_d, blocks * sizeof(int)));

    GPU_ERR_CHK(cudaMemcpy(a_d, a, sizeof(int) * size,
                cudaMemcpyHostToDevice));
    GPU_ERR_CHK(cudaMemcpy(b_d, b, sizeof(int) * size,
                cudaMemcpyHostToDevice));

    //cu_dot<BLOCK_SIZE> <<<dimgrid, dimblock, cache_size>>>(a_d, b_d, block_results_d, size);
    cu_dot<BLOCK_SIZE> <<<blocks, BLOCK_SIZE, cache_size>>>(a_d, b_d, block_results_d, size);

    GPU_ERR_CHK(cudaMemcpy(block_results, block_results_d,
                blocks * sizeof(int), cudaMemcpyDeviceToHost));

    for(int z = 0; z < blocks; z++) {
        *result += block_results[z];
    }

    //*result = *block_results;

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(block_results_d);
}
