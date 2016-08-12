// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <string.h>

//inicio simple GA
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <iomanip>
# include <cmath>
# include <cstring>

using namespace std;

//  Static arameters
# define POPSIZE 50
# define MAXGENS 100
# define NVARS 3
# define PXOVER 0.8
# define PMUTATION 0.15
//
//  Each GENOTYPE is a member of the population, with
//  gene: a string of variables,
//  fitness: the fitness
//  upper: the variable upper bounds,
//  lower: the variable lower bounds,
//  rfitness: the relative fitness,
//  cfitness: the cumulative fitness.
//
struct genotype
{
  double gene[NVARS];
  double fitness;
  double upper[NVARS];
  double lower[NVARS];
  double rfitness;
  double cfitness;
};

struct genotype population[POPSIZE+1];
struct genotype newpopulation[POPSIZE+1];

void crossover ( int &seed );
void elitist ( );
void evaluate ( );
int i4_uniform_ab ( int a, int b, int &seed );
void initialize ( int &seed );
void keep_the_best ( );
void mutate ( int &seed );
double r8_uniform_ab ( double a, double b, int &seed );
void report ( int generation );
void selector ( int &seed );
void Xover ( int one, int two, int &seed );


//****************************************************************************80

void crossover ( int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    CROSSOVER selects two parents for the single point crossover.
//
//  Local parameters:
//
//    Local, int FIRST, is a count of the number of members chosen.
//
//  Parameters:
//
//    Input/output, int &SEED, a seed for the random number generator.
//
{
  const double a = 0.0;
  const double b = 1.0;
  int mem;
  int one;
  int first = 0;
  double x;

  for ( mem = 0; mem < POPSIZE; ++mem )
  {
    x = r8_uniform_ab ( a, b, seed );

    if ( x < PXOVER )
    {
      ++first;

      if ( first % 2 == 0 )
      {
        Xover ( one, mem, seed );
      }
      else
      {
        one = mem;
      }

    }
  }
  return;
}
//****************************************************************************80

void elitist ( )

//****************************************************************************80
//
//  Purpose:
//
//    ELITIST stores the best member of the previous generation.
//
//  Discussion:
//
//    The best member of the previous generation is stored as
//    the last in the array. If the best member of the current
//    generation is worse then the best member of the previous
//    generation, the latter one would replace the worst member
//    of the current population.
//
//  Local parameters:
//
//    Local, double BEST, the best fitness value.
//
//    Local, double WORST, the worst fitness value.
//
{
  int i;
  double best;
  int best_mem;
  double worst;
  int worst_mem;

  best = population[0].fitness;
  worst = population[0].fitness;

  for ( i = 0; i < POPSIZE - 1; ++i )
  {
    if ( population[i+1].fitness < population[i].fitness )
    {

      if ( best <= population[i].fitness )
      {
        best = population[i].fitness;
        best_mem = i;
      }

      if ( population[i+1].fitness <= worst )
      {
        worst = population[i+1].fitness;
        worst_mem = i + 1;
      }

    }
    else
    {

      if ( population[i].fitness <= worst )
      {
        worst = population[i].fitness;
        worst_mem = i;
      }

      if ( best <= population[i+1].fitness )
      {
        best = population[i+1].fitness;
        best_mem = i + 1;
      }

    }

  }
//
//  If the best individual from the new population is better than
//  the best individual from the previous population, then
//  copy the best from the new population; else replace the
//  worst individual from the current population with the
//  best one from the previous generation
//
  if ( population[POPSIZE].fitness <= best )
  {
    for ( i = 0; i < NVARS; i++ )
    {
      population[POPSIZE].gene[i] = population[best_mem].gene[i];
    }
    population[POPSIZE].fitness = population[best_mem].fitness;
  }
  else
  {
    for ( i = 0; i < NVARS; i++ )
    {
      population[worst_mem].gene[i] = population[POPSIZE].gene[i];
    }
    population[worst_mem].fitness = population[POPSIZE].fitness;
  }

  return;
}
//****************************************************************************80

void evaluate ( )

//****************************************************************************80
//
//  Purpose:
//
//    EVALUATE implements the user-defined valuation function
//
//  Discussion:
//
//    Each time this is changed, the code has to be recompiled.
//    The current function is:  x[1]^2-x[1]*x[2]+x[3]
//
//
{
  int member;
  int i;
  double x[NVARS+1];

  for ( member = 0; member < POPSIZE; member++ )
  {
    for ( i = 0; i < NVARS; i++ )
    {
      x[i+1] = population[member].gene[i];
    }
    population[member].fitness = ( x[1] * x[1] ) - ( x[1] * x[2] ) + x[3];
  }
  return;
}
//****************************************************************************80

int i4_uniform_ab ( int a, int b, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    I4_UNIFORM_AB returns a scaled pseudorandom I4 between A and B.
//
//  Discussion:
//
//    The pseudorandom number should be uniformly distributed
//    between A and B.
//
//
//  Parameters:
//
//    Input, int A, B, the limits of the interval.
//
//    Input/output, int &SEED, the "seed" value, which should NOT be 0.
//    On output, SEED has been updated.
//
//    Output, int I4_UNIFORM, a number between A and B.
//
{
  int c;
  const int i4_huge = 2147483647;
  int k;
  float r;
  int value;

  if ( seed == 0 )
  {
    cerr << "\n";
    cerr << "I4_UNIFORM_AB - Fatal error!\n";
    cerr << "  Input value of SEED = 0.\n";
    exit ( 1 );
  }
//
//  Guarantee A <= B.
//
  if ( b < a )
  {
    c = a;
    a = b;
    b = c;
  }

  k = seed / 127773;

  seed = 16807 * ( seed - k * 127773 ) - k * 2836;

  if ( seed < 0 )
  {
    seed = seed + i4_huge;
  }

  r = ( float ) ( seed ) * 4.656612875E-10;
//
//  Scale R to lie between A-0.5 and B+0.5.
//
  r = ( 1.0 - r ) * ( ( float ) a - 0.5 )
    +         r   * ( ( float ) b + 0.5 );
//
//  Use rounding to convert R to an integer between A and B.
//
  value = round ( r );
//
//  Guarantee A <= VALUE <= B.
//
  if ( value < a )
  {
    value = a;
  }
  if ( b < value )
  {
    value = b;
  }

  return value;
}
//****************************************************************************80

void initialize ( int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    INITIALIZE initializes the genes within the variables bounds.
//
//  Discussion:
//
//    It also initializes (to zero) all fitness values for each
//    member of the population. It reads upper and lower bounds
//    of each variable from the input file `gadata.txt'. It
//    randomly generates values between these bounds for each
//    gene of each genotype in the population. The format of
//    the input file `gadata.txt' is
//
//      var1_lower_bound var1_upper bound
//      var2_lower_bound var2_upper bound ...
//
//
//  Parameters:
//
//    Input, string FILENAME, the name of the input file.
//
//    Input/output, int &SEED, a seed for the random number generator.
//
{
  int i;
  int j;
  double lbound; // mínimo
  double ubound; // máximo

  // Restricciones
  double matriz_entrada[2][NVARS];
  matriz_entrada[0][0]=0.0;
  matriz_entrada[1][0]=5.0;

  matriz_entrada[0][1]=0.0;
  matriz_entrada[1][1]=5.0;

  matriz_entrada[0][2]=-2.0;
  matriz_entrada[1][2]=2.0;

//
//  Initialize variables within the bounds
//
  for ( i = 0; i < NVARS; i++ )
  {

    lbound=matriz_entrada[0][i];
    ubound=matriz_entrada[1][i];

    for ( j = 0; j < POPSIZE; j++ )
    {
      population[j].fitness = 0;
      population[j].rfitness = 0;
      population[j].cfitness = 0;
      population[j].lower[i] = lbound;
      population[j].upper[i]= ubound;
      population[j].gene[i] = r8_uniform_ab ( lbound, ubound, seed );
    }
  }

  return;
}
//****************************************************************************80

void keep_the_best ( )

//****************************************************************************80
//
//  Purpose:
//
//    KEEP_THE_BEST keeps track of the best member of the population.
//
//  Discussion:
//
//    Note that the last entry in the array Population holds a
//    copy of the best individual.
//
//  Local parameters:
//
//    Local, int CUR_BEST, the index of the best individual.
//
{
  int cur_best;
  int mem;
  int i;

  cur_best = 0;

  for ( mem = 0; mem < POPSIZE; mem++ )
  {
    if ( population[POPSIZE].fitness < population[mem].fitness )
    {
      cur_best = mem;
      population[POPSIZE].fitness = population[mem].fitness;
    }
  }
//
//  Once the best member in the population is found, copy the genes.
//
  for ( i = 0; i < NVARS; i++ )
  {
    population[POPSIZE].gene[i] = population[cur_best].gene[i];
  }

  return;
}
//****************************************************************************80

void mutate ( int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    MUTATE performs a random uniform mutation.
//
//  Discussion:
//
//    A variable selected for mutation is replaced by a random value
//    between the lower and upper bounds of this variable.
//
//  Parameters:
//
//    Input/output, int &SEED, a seed for the random number generator.
//
{
  const double a = 0.0;
  const double b = 1.0;
  int i;
  int j;
  double lbound;
  double ubound;
  double x;

  for ( i = 0; i < POPSIZE; i++ )
  {
    for ( j = 0; j < NVARS; j++ )
    {
      x = r8_uniform_ab ( a, b, seed );
      if ( x < PMUTATION )
      {
        lbound = population[i].lower[j];
        ubound = population[i].upper[j];
        population[i].gene[j] = r8_uniform_ab ( lbound, ubound, seed );
      }
    }
  }

  return;
}
//****************************************************************************80

double r8_uniform_ab ( double a, double b, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    R8_UNIFORM_AB returns a scaled pseudorandom R8.
//
//  Discussion:
//
//    The pseudorandom number should be uniformly distributed
//    between A and B.
//
//  Parameters:
//
//    Input, double A, B, the limits of the interval.
//
//    Input/output, int &SEED, the "seed" value, which should NOT be 0.
//    On output, SEED has been updated.
//
//    Output, double R8_UNIFORM_AB, a number strictly between A and B.
//
{
  int i4_huge = 2147483647;
  int k;
  double value;

  if ( seed == 0 )
  {
    cerr << "\n";
    cerr << "R8_UNIFORM_AB - Fatal error!\n";
    cerr << "  Input value of SEED = 0.\n";
    exit ( 1 );
  }

  k = seed / 127773;

  seed = 16807 * ( seed - k * 127773 ) - k * 2836;

  if ( seed < 0 )
  {
    seed = seed + i4_huge;
  }

  value = ( double ) ( seed ) * 4.656612875E-10;

  value = a + ( b - a ) * value;

  return value;
}
//****************************************************************************80

void report ( int generation )

//****************************************************************************80
//
//  Purpose:
//
//    REPORT reports progress of the simulation.
//
//
//  Local parameters:
//
//    Local, double avg, the average population fitness.
//
//    Local, best_val, the best population fitness.
//
//    Local, double square_sum, square of sum for std calc.
//
//    Local, double stddev, standard deviation of population fitness.
//
//    Local, double sum, the total population fitness.
//
//    Local, double sum_square, sum of squares for std calc.
//
{
  double avg;
  double best_val;
  int i;
  double square_sum;
  double stddev;
  double sum;
  double sum_square;

  if ( generation == 0 )
  {
    cout << "\n";
    cout << "  Generation       Best            Average       Standard \n";
    cout << "  number           value           fitness       deviation \n";
    cout << "\n";
  }

  sum = 0.0;
  sum_square = 0.0;

  for ( i = 0; i < POPSIZE; i++ )
  {
    sum = sum + population[i].fitness;
    sum_square = sum_square + population[i].fitness * population[i].fitness;
  }

  avg = sum / ( double ) POPSIZE;
  square_sum = avg * avg * POPSIZE;
  stddev = sqrt ( ( sum_square - square_sum ) / ( POPSIZE - 1 ) );
  best_val = population[POPSIZE].fitness;

  cout << "  " << setw(8) << generation
       << "  " << setw(14) << best_val
       << "  " << setw(14) << avg
       << "  " << setw(14) << stddev << "\n";

  return;
}
//****************************************************************************80

void selector ( int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    SELECTOR is the selection function.
//
//  Discussion:
//
//    Standard proportional selection for maximization problems incorporating
//    the elitist model.  This makes sure that the best member always survives.
//
//
//  Parameters:
//
//    Input/output, int &SEED, a seed for the random number generator.
//
{
  const double a = 0.0;
  const double b = 1.0;
  int i;
  int j;
  int mem;
  double p;
  double sum;
//
//  Find the total fitness of the population.
//
  sum = 0.0;
  for ( mem = 0; mem < POPSIZE; mem++ )
  {
    sum = sum + population[mem].fitness;
  }
//
//  Calculate the relative fitness of each member.
//
  for ( mem = 0; mem < POPSIZE; mem++ )
  {
    population[mem].rfitness = population[mem].fitness / sum;
  }
//
//  Calculate the cumulative fitness.
//
  population[0].cfitness = population[0].rfitness;
  for ( mem = 1; mem < POPSIZE; mem++ )
  {
    population[mem].cfitness = population[mem-1].cfitness +
      population[mem].rfitness;
  }
//
//  Select survivors using cumulative fitness.
//
  for ( i = 0; i < POPSIZE; i++ )
  {
    p = r8_uniform_ab ( a, b, seed );
    if ( p < population[0].cfitness )
    {
      newpopulation[i] = population[0];
    }
    else
    {
      for ( j = 0; j < POPSIZE; j++ )
      {
        if ( population[j].cfitness <= p && p < population[j+1].cfitness )
        {
          newpopulation[i] = population[j+1];
        }
      }
    }
  }
//
//  Overwrite the old population with the new one.
//
  for ( i = 0; i < POPSIZE; i++ )
  {
    population[i] = newpopulation[i];
  }

  return;
}

//****************************************************************************80

void Xover ( int one, int two, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    XOVER performs crossover of the two selected parents.
//
//
//  Local parameters:
//
//    Local, int point, the crossover point.
//
//  Parameters:
//
//    Input, int ONE, TWO, the indices of the two parents.
//
//    Input/output, int &SEED, a seed for the random number generator.
//
{
  int i;
  int point;
  double t;
//
//  Select the crossover point.
//
  point = i4_uniform_ab ( 0, NVARS - 1, seed );
//
//  Swap genes in positions 0 through POINT-1.
//
  for ( i = 0; i < point; i++ )
  {
    t                       = population[one].gene[i];
    population[one].gene[i] = population[two].gene[i];
    population[two].gene[i] = t;
  }

  return;
}


/**
 * Función Ackley
 *
 */
template <int BLOCK_SIZE> __global__ void
matrixAckleyCUDA(float *C, float *A, int wA, float valorA, float valorB, float valorC)
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
    float Csub1 = 0;
    float Csub2 = 0;

    float term1 = 0;
    float term2 = 0;

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
        	Csub1 += As[ty][k]*As[ty][k];
            Csub2 += cos(valorC*As[ty][k]);
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

    term1= (0-valorA)*exp( (0-valorB)*sqrt(Csub1/wA) );
    term2= 0-exp(Csub2/wA);

    C[c + wA * ty + tx] = term1 + term2 + valorA + exp(1.0);

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

/*
    //  CUDA kernel
    if (block_size == 16)
    {
        matrixAckleyCUDA<16><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
    }
    else
    {
        matrixAckleyCUDA<32><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
    }
*/

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
		matrixAckleyCUDA<16><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
	}
	else
	{
		matrixAckleyCUDA<32><<< grid, threads >>>(d_C, d_A, dimsA.x, valorA, valorB, valorC);
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
        msecTotal,
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
	//  Start simple GA

	//  Discussion:
	//    Each generation involves selecting the best
	//    members, performing crossover & mutation and then
	//    evaluating the resulting population, until the terminating
	//    condition is satisfied
	//
	//    This is a simple genetic algorithm implementation where the
	//    evaluation function takes positive values only and the
	//    fitness of an individual is the same as the value of the
	//    objective function.
	//
	//  Parameters:
	//    MAXGENS is the maximum number of generations.
	//    NVARS is the number of problem variables.
	//    PMUTATION is the probability of mutation.
	//    POPSIZE is the population size.
	//    PXOVER is the probability of crossover.

	int generation;
	int i;
	int seed;

	cout << "\n";

	if ( NVARS < 2 )
	{
	  cout << "\n";
	  cout << "  The crossover modification will not be available,\n";
	  cout << "  since it requires 2 <= NVARS.\n";
	}

	seed = 123456789;

	initialize ( seed );

	evaluate ( );

    keep_the_best ( );

	for ( generation = 0; generation < MAXGENS; generation++ )
	{
	  selector ( seed );
	  crossover ( seed );
	  mutate ( seed );
	  report ( generation );
	  evaluate ( );
	  elitist ( );
	}

	cout << "\n";
	cout << " Best member after " << MAXGENS << " generations:\n";
	cout << "\n";

	for ( i = 0; i < NVARS; i++ )
	{
     cout << "  var(" << i << ") = " << population[POPSIZE].gene[i] << "\n";
	}

	cout << "\n";
	cout << "  Best fitness = " << population[POPSIZE].fitness << "\n";
	//
	//  Terminate simple GA


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
