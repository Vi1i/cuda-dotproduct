#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

extern void dot_product(int * result, int * a, int * b,
        size_t size);
extern void gen_curand_array(unsigned long long seed, int * array,
        size_t array_size);

void dot(int * result, int * a, int * b, size_t size) {
    int z;
    for(z = 0; z < size; z++) {
        *result += a[z] * b[z];
    }
}

int main(int argc, char * argv[]) {
    if(argc != 2) {
        fprintf(stderr, "%s [SIZE]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    size_t size = (int) strtol(argv[1], (char **)NULL, 10);
    int * a = (int *) malloc(size * sizeof(int));
    int * b = (int *) malloc(size * sizeof(int));
    int * result = (int *) malloc(sizeof(int));
    int * result_seq = (int *) malloc(sizeof(int));
    unsigned long long seed_a = 1;
    unsigned long long seed_b = 2;
    struct timeval start;
    struct timeval end;
    int time_seq;
    int time_par;

    gen_curand_array(seed_a, a, size);

    gen_curand_array(seed_b, b, size);

    *result_seq = 0;
    *result = 0;

    gettimeofday(&start, NULL);
    dot_product(result, a, b, size);
    gettimeofday(&end, NULL);
    time_seq = (end.tv_sec  - start.tv_sec) * 1000000L +
        (end.tv_usec - start.tv_usec);

    gettimeofday(&start, NULL);
    dot(result_seq, a, b, size);
    gettimeofday(&end, NULL);
    time_par = (end.tv_sec  - start.tv_sec) * 1000000L +
        (end.tv_usec - start.tv_usec);

    printf("GPU : %d : %d \n", time_par, *result);
    printf("CPU : %d : %d \n", time_seq, *result_seq);

    free(a);
    free(b);
    exit(EXIT_SUCCESS);
}
