/**
 * File: quamsimV1.cu
 * Author: Sirajus Salekin
 *
 * Contains cuda kernel that performs matrix multiplication to simulate
 * applying quantum gate on an n-qubit quantum circuit.
 *
 * https://dl.acm.org/doi/abs/10.1145/3447818.3460357
 *  
 * See section 2 in this paper for details. Motivation behind the algorithm
 * implemented here is directly quoted from the paper:
 *
 * "An n-qubit system is represented by 2^n complex numbers. A gate on
 * the n-qubit system is a 2^n x 2^n unitary matrix. However, the 2^n x 2^n 
 * matrix representations of single qubit gates and controlled gates are
 * very sparse and there is no need to construct the matrix. Instead,
 * these gates can be implemented by several small matrix multiplications.
 * ...
 * And applying a two-qubit controlled gate on the n-qubit system
 * is equivalent to 2^(n−2) matrix multiplication tasks. Each task updates
 * two positions with the index of the control qubit C equals 1 and
 * only the target qubit t differs."
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


/**
 * CUDA Kernel Device code
 *
 * The kernel code takes the thread id and use it to calculate the pair of elements
 * in the quantum state that should be computed together. It checks for 1 or 0 at the "t-th" bit
 * and calculates the matrix once it finds that. However, when it figures out that pairing
 * criteria does not match, that thread does not do anything. Some optimization that can be done here
 *
 * Inputs:
 *  1. float *A: statevector of the n-qubit system
 *  2. float *C: statevector after applying the quantum gate operation 
 *  3. float a, b, c, d: represent the quantum gate being applied to
 *     |a b|
 *     |c d|
 *  4. state_size: size of the statevector pointed by *A and *C
 *  5. t-bit: The bit where the target qubit differs.
 */

__global__ void matrix_mul(float *A, float *C, float a, float b, float c, float d, int state_size, int t_bit){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //i=x1, flipped=x2
    int flipped = ((1 << t_bit) | i);
    if (i < state_size){
        if (flipped  > i){
            C[i] = (A[i] * a ) + (A[flipped] * b);
            C[flipped] = (A[i] * c ) + (A[flipped] * d);
        }
    }
}


/**
 * Host main routine
 */
int main(int argc, char **argv){
    /*
        INPUT READING FROM FILE
    */
     float a, b, c, d; // quantum gate matrix
     int t_bit;
     int vector_size=pow(2,30), final_size;
     float* state_vector  = (float*)malloc(vector_size * sizeof(float));

     float number;
     FILE* in_file = fopen(argv[1], "r"); // read only

     if (!in_file ){// equivalent to saying if ( in_file == NULL )
        printf("oops, file can't be read\n");
        exit(-1);
     }

     // attempt to read the next line and store
     // the value in the "number" variable
     int count = 0;
     float last;
     while ( fscanf(in_file, "%f", &number ) != EOF ){
          if (count==0) a = number;
          else if (count==1) b = number;
          else if (count==2) c = number;
          else if (count==3) d = number;

          else {
              state_vector[count-4] = number;

           }

          last = number;
          count++;
        }

     t_bit = (int) last;
     final_size = count - 5;
     //printf("Qubit %d size: %d a:%f, b:%f c:%f d:%f\n", qubit, final_size, a, b, c, d);
     /* END READING FROM INPUT */


     // Error code to check return values for CUDA calls
     cudaError_t err = cudaSuccess;

     // Print the vector length to be used, and compute its size
     int numElements = final_size;
     size_t size = numElements * sizeof(float);

     // Allocate the host input vector A
     float *h_A = (float *)malloc(size);

     // Allocate the host output vector C
     float *h_C = (float *)malloc(size);

     // Verify that allocations succeeded
     if (h_A == NULL || h_C == NULL){
         fprintf(stderr, "Failed to allocate host vectors!\n");
         exit(EXIT_FAILURE);
     }

     // Initialize the host input vectors
     for (int i = 0; i < numElements; ++i){
         h_A[i] = state_vector[i];
     }

     // Allocate the device input vector A
     float *d_A = NULL;
     err = cudaMalloc((void **)&d_A, size);

     if (err != cudaSuccess){
         fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }


     // Allocate the device output vector C
     float *d_C = NULL;
     err = cudaMalloc((void **)&d_C, size);

     if (err != cudaSuccess){
         fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     // Copy the host input vectors A and B in host memory to the device input vectors in
     // device memory
     err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

     if (err != cudaSuccess){
         fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }


     // Launch the CUDA Kernel
     int threadsPerBlock = 256;
     int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
     //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
     matrix_mul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, a, b, c, d, numElements, t_bit);
     err = cudaGetLastError();

     if (err != cudaSuccess){
         fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     // Copy the device result vector in device memory to the host result vector
     // in host memory.
     err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
     if (err != cudaSuccess){
         fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     // print out the results stored in h_C
     for (int i = 0; i < numElements; ++i){
         printf("%.3f\n", h_C[i]);
     }


     // Free device global memory
     err = cudaFree(d_A);
     if (err != cudaSuccess){
         fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }


     err = cudaFree(d_C);
     if (err != cudaSuccess){
         fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     // Free host memory
     free(h_A);
     free(h_C);

     //Free IO memory
     free(state_vector);

     // Reset the device and exit
     // cudaDeviceReset causes the driver to clean up all state. While
     // not mandatory in normal operation, it is good practice.  It is also
     // needed to ensure correct operation when the application is being
     // profiled. Calling cudaDeviceReset causes all profile data to be
     // flushed before the application exits
     err = cudaDeviceReset();

     if (err != cudaSuccess){
         fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     return 0;
}

