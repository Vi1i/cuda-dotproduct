/*
 * blockAndthread.c
 * A "driver" program that calls a routine (i.e. a kernel)
 * that executes on the GPU.  The kernel fills two int arrays
 * with the block ID and the thread ID
 *
 * Note: the kernel code is found in the file 'blockAndThread.cu'
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc blockAndThread.c blockAndThread.cu
 */

#include <stdio.h>
#define SIZEOFARRAY 64 

// The function fillArray is in the file blockAndThread.cu
extern void fillArray(int *blockId, int *threadId, int size);

int main (int argc, char *argv[])
{
    // Declare arrays and initialize to 0
    int blockId[SIZEOFARRAY];
    int threadId[SIZEOFARRAY];
    int i;
    for (i=0; i < SIZEOFARRAY; i++) {
        blockId[i]=0;
        threadId[i]=0;
    }

    // Print the initial arrays
    printf ("Initial state of the array blockId:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", blockId[i]);
    }
    printf ("\n");
    printf ("Initial state of the array threadId:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", threadId[i]);
    }
    printf ("\n");

    // Call the function that will call the GPU function
    fillArray (blockId, threadId, SIZEOFARRAY);

    // Again, print the arrays
    printf ("Final state of the array blockId:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", blockId[i]);
    }
    printf ("\n");
    printf ("Initial state of the array threadId:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", threadId[i]);
    }
    printf ("\n");

    return 0;
}
