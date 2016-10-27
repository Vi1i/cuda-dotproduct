/*
 * simple.c
 * A simple "driver" program that calls a routine (i.e. a kernel)
 * that executes on the GPU.  The kernel uses k threads to fill
 * an array of integers with the consecutive values 0,1,2,..., k-1.
 *
 * Note: the kernel code is found in the file 'simple.cu'
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc simple.c simple.cu
 */

#include <stdio.h>
#define SIZEOFARRAY 64 

// The function fillArray is in the file simple.cu
extern void fillArray(int *a, int size);

int main (int argc, char *argv[])
{
    // Declare the array and initialize to 0
    int a[SIZEOFARRAY];
    int i;
    for (i=0; i < SIZEOFARRAY; i++) {
        a[i]=0;
    }

    // Print the initial array
    printf ("Initial state of the array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", a[i]);
    }
    printf ("\n");

    // Call the function that will call the GPU function
    fillArray (a,SIZEOFARRAY);

    // Again, print the array
    printf ("Final state of the array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", a[i]);
    }
    printf ("\n");

    return 0;
}
