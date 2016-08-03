// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#define _USE_MATH_DEFINES
#include <math.h>

/**
 * Función Ackley
 *
 */
template <int BLOCK_SIZE> __global__ void
matrixAckelyCUDA(float *C, float *A, int wA, float valorA, float valorB, float valorC)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;


    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;


    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin;
         a <= aEnd;
         a += aStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        //__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        //Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Mulatoria the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            //Csub += As[ty][k] * Bs[k][tx];
            Csub += As[ty][k];

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    //int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    //C[c + wB * ty + tx] = Csub;
    int c = wA * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    float aux = valorA + valorB + valorC;
    C[c + wA * ty + tx] = Csub ;//+ aux;

}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixAckley(int argc, char **argv, int block_size, dim3 &dimsA, float valor, float valorA, float valorB, float valorC)
{
    // Allocate host memory for matrices A
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    // Initialize host memory
    //constantInit(h_A, size_A, 1.0f);
    constantInit(h_A, size_A, valor);

    // Allocate device memory
    float *d_A,  *d_C;

/*
	unsigned int mem_size_C = dimsA.y  * sizeof(float);
	// Allocate host vector C
    float *h_C = (float *)malloc(mem_size_C);
*/
    // Allocate host matrix C
    dim3 dimsC(dimsA.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.x * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }


    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsA.x / threads.x, dimsA.y / threads.y);


    // Performs warmup operation using matrixSum CUDA kernel
    if (block_size == 16)
    {
        matrixAckelyCUDA<16><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
    }
    else
    {
        matrixAckelyCUDA<32><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
    }


    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
	if (block_size == 16)
	{
		matrixAckelyCUDA<16><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
	}
	else
	{
		matrixAckelyCUDA<32><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
	}


    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPermatrixSum = msecTotal ;
    double flopsPermatrixSum = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsA.x;
    double gigaFlops = (flopsPermatrixSum * 1.0e-9f) / (msecPermatrixSum / 1000.0f);
    printf(
        " \"datos_computo\" : { \n   \"performance\" : \"%.2f GFlop/s\", \n   \"time\" : \"%.3f msec\", \n   \"size\" : \"%.0f Ops\", \n   \"workgroupSize\" : \"%u threads/block\" \n }, \n",
        gigaFlops,
        msecPermatrixSum,
        flopsPermatrixSum,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf(" \"num_resultados\" : \"%u\", \n",dimsA.y);
    printf(" \"resultados\" : { \n");


    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {

    	if(i%dimsC.x==0){
    		printf("   \"calculo fila %d\" : \"%.8f\", \n", i/dimsC.x, h_C[i]);
    	}

    }

    printf("   \"x\": \"x\" \n }");

    // Clean up memory
    free(h_A);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_C);


    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return EXIT_SUCCESS;

}


/**
 * Program main
 */
int main(int argc, char **argv)
{

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -width=Width's matrix -height=Height's matrix \n");
        printf("      -cvalue=contant value (constant value to initialize the matrix)\n");
        printf("      -a= (20 by default) -b= (0.2 by default) -c= (2*PI by default) \n");


        exit(EXIT_SUCCESS);
    }

	// Salida del programa en formato JSON
	printf("{ \"calculo\":{ \n");
    printf(" \"nombre\" : \"Ackley function in matrix using CUDA\", \n");

    // valor de las filas de la matriz (1 por defecto)
    float valor=1.0;
    if (checkCmdLineFlag(argc, (const char **)argv, "cvalue"))
    {
        valor = getCmdLineArgumentInt(argc, (const char **)argv, "cvalue");
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf(" \"dispositivo\" : \"GPU Device %d: '%s' with compute capability %d.%d\", \n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    dim3 dimsA(block_size, block_size, 1);

    // width of Matrix
    if (checkCmdLineFlag(argc, (const char **)argv, "width"))
    {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    }

    // height of Matrix
    if (checkCmdLineFlag(argc, (const char **)argv, "height"))
    {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }

    float valorA=20;
    if (checkCmdLineFlag(argc, (const char **)argv, "a"))
    {
    	valorA = getCmdLineArgumentInt(argc, (const char **)argv, "a");
    }

    float valorB=0.2;
    if (checkCmdLineFlag(argc, (const char **)argv, "b"))
    {
        valorB = getCmdLineArgumentInt(argc, (const char **)argv, "b");
    }

    float valorC=2*M_PI;
    if (checkCmdLineFlag(argc, (const char **)argv, "c"))
    {
        valorC = getCmdLineArgumentInt(argc, (const char **)argv, "c");
    }


    printf(" \"info_matriz\" : \"Matrix(%d,%d) with constant value %f\", \n", dimsA.x, dimsA.y, valor);
    printf(" \"info_input\" : { \n   \"matriz\" : \" * \", \n   \"a\" : \"%f\", \n   \"b\" : \"%f\", \n   \"c\" : \"%f\" \n }, \n", valorA, valorB, valorC);

    int matrix_result = matrixAckley(argc, argv, block_size, dimsA, valor, valorA, valorB, valorC);

    printf("\n} \n}");

    exit(matrix_result);

}
