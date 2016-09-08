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

// Parameters
int POPSIZE = 50; 		//    POPSIZE is the population size.
int MAXGENS = 100;  	//    MAXGENS is the maximum number of generations.
int NVARS = 10;  		//    NVARS is the number of problem variables.
float MIN = -32.768;	//	  MIN is the minimum value to genotype
float MAX = 32.768;		//	  MAX is the maximum value to genotype
float PXOVER = 0.8; 	//    PXOVER is the probability of crossover.
float PMUTATION = 0.15; //    PMUTATION is the probability of mutation.


//  Static parameters
const int MAXVARS = 1000;
const int MAXPOPSIZE = 10000;

//
//  Each GENOTYPE is a member of the population, with
//  gene: a string of variables,
//  fitness: the fitness
//  upper: the variable upper bounds,
//  lower: the variable lower bounds,
//  rfitness: the relative fitness,
//  cfitness: the cumulative fitness.
//
struct genotype{
  double gene[MAXVARS];
  double fitness;
  double upper[MAXVARS];
  double lower[MAXVARS];
  double rfitness;
  double cfitness;
};

struct genotype population[MAXPOPSIZE];
struct genotype newpopulation[MAXPOPSIZE];

void crossover ( int &seed );
void elitist ( );
int i4_uniform_ab ( int a, int b, int &seed );
void initialize ( int &seed );
void keep_the_best ( );
void mutate ( int &seed );
double r8_uniform_ab ( double a, double b, int &seed );
void report ( int generation );
void selector ( int &seed );
void Xover ( int one, int two, int &seed );
void resultToHost(float *data, int size);
void init(float *data);


/*
 * crossover: selects two parents for the single point crossover
 *
 * FIRST is a count of the number of members chosen.
 * Input/output, int &SEED, a seed for the random number generator.
 *
 * */
void crossover ( int &seed ){

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

/*
 * elitist: stores the best member of the previous generation
 *
 * (The best member of the previous generation is stored as the last in the array)
 *
 * Local parameters: BEST (best fitness value) WORST (worst fitness value)
 */
void elitist ( ){

  int i;
  double best;
  int best_mem;
  double worst;
  int worst_mem;

  best = population[0].fitness;
  worst = population[0].fitness;

  for ( i = 0; i < POPSIZE - 1; ++i ){
    if ( population[i+1].fitness > population[i].fitness ){

      if ( best >= population[i].fitness ){
        best = population[i].fitness;
        best_mem = i;
      }

      if ( population[i+1].fitness >= worst ){
        worst = population[i+1].fitness;
        worst_mem = i + 1;
      }

    }
    else{

      if ( population[i].fitness >= worst ){
        worst = population[i].fitness;
        worst_mem = i;
      }

      if ( best >= population[i+1].fitness ){
        best = population[i+1].fitness;
        best_mem = i + 1;
      }

    }

  }

//  If the best individual from the new population is better than
//  the best individual from the previous population, then
//  copy the best from the new population; else replace the
//  worst individual from the current population with the
//  best one from the previous generation
  if ( population[POPSIZE].fitness > best ){
    for ( i = 0; i < NVARS; i++ ){
      population[POPSIZE].gene[i] = population[best_mem].gene[i];
    }
    population[POPSIZE].fitness = population[best_mem].fitness;
  }
  else{
    for ( i = 0; i < NVARS; i++ ){
      population[worst_mem].gene[i] = population[POPSIZE].gene[i];
    }
    population[worst_mem].fitness = population[POPSIZE].fitness;
  }

  return;
}


/*
 * i4_uniform_ab: returns a scaled pseudorandom I4 between A and B
 *
 * (The pseudorandom number should be uniformly distributed between A and B)
 *
 * Input: A, B, the limits of the interval
 *
 * Output: I4_UNIFORM (a number between A and B)
 */
int i4_uniform_ab ( int a, int b, int &seed ){

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

  //  Guarantee A <= B.
  if ( b < a ){
    c = a;
    a = b;
    b = c;
  }

  k = seed / 127773;

  seed = 16807 * ( seed - k * 127773 ) - k * 2836;

  if ( seed < 0 ){
    seed = seed + i4_huge;
  }

  r = ( float ) ( seed ) * 4.656612875E-10;

  //  Scale R to lie between A-0.5 and B+0.5.
  r = ( 1.0 - r ) * ( ( float ) a - 0.5 )
    +         r   * ( ( float ) b + 0.5 );

  //  Use rounding to convert R to an integer between A and B.
  value = round ( r );

  //  Guarantee A <= VALUE <= B.
  if ( value < a ){
    value = a;
  }
  if ( b < value ){
    value = b;
  }

  return value;
}



/*
 * initialize: initializes the genes within the variables bounds
 *
 * Input: FILENAME, the name of the input file
 * Input/output: int &SEED, a seed for the random number generator
 */
void initialize ( int &seed ){

  int i;
  int j;
  double lbound = MIN;
  double ubound = MAX;

  //  Initialize variables within the bounds
  for ( i = 0; i < NVARS; i++ ){
    for ( j = 0; j < POPSIZE; j++ ){
      population[j].fitness = 10;
      population[j].rfitness = 10;
      population[j].cfitness = 10;
      population[j].lower[i] = lbound;
      population[j].upper[i]= ubound;
      population[j].gene[i] = r8_uniform_ab ( lbound, ubound, seed );
      //printf("(%d)%f",j,population[j].gene[i]);
    }
  }
  return;
}


/*
 * keep_the_best: keeps track of the best member of the population
 *
 * (Note that the last entry in the array Population holds a copy of the best individual)
 *
 * Local parameters: CUR_BEST, the index of the best individual
 */
void keep_the_best ( ){

  int cur_best;
  int mem;
  int i;

  cur_best = 0;

  for ( mem = 0; mem < POPSIZE; mem++ ){
    if ( population[POPSIZE].fitness > population[mem].fitness ){
      cur_best = mem;
      population[POPSIZE].fitness = population[mem].fitness;
    }
  }

  //  Once the best member in the population is found, copy the genes.
  for ( i = 0; i < NVARS; i++ ){
    population[POPSIZE].gene[i] = population[cur_best].gene[i];
  }

  return;
}

/*
 * mutate: performs a random uniform mutation
 *
 * (A variable selected for mutation is replaced by a random value between the lower and upper bounds of this variable)
 *
 * Input/output, int &SEED, a seed for the random number generator
 */
void mutate ( int &seed ){

  const double a = 0.0;
  const double b = 1.0;
  int i;
  int j;
  double lbound;
  double ubound;
  double x;

  for ( i = 0; i < POPSIZE; i++ ){
    for ( j = 0; j < NVARS; j++ ){
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


/*
 * r8_uniform_ab: returns a scaled pseudorandom R8
 *
 * ( The pseudorandom number should be uniformly distributed between A and B)
 *
 * Input: A, B, the limits of the interval
 * Input/output:&SEED, the "seed" value, which should NOT be 0. On output, SEED has been updated
 * Output: R8_UNIFORM_AB, a number strictly between A and B
 */
double r8_uniform_ab ( double a, double b, int &seed ){

  int i4_huge = 2147483647;
  int k;
  double value;

  if ( seed == 0 ){
    cerr << "\n";
    cerr << "R8_UNIFORM_AB - Fatal error!\n";
    cerr << "  Input value of SEED = 0.\n";
    exit ( 1 );
  }

  k = seed / 127773;

  seed = 16807 * ( seed - k * 127773 ) - k * 2836;

  if ( seed < 0 ){
    seed = seed + i4_huge;
  }

  value = ( double ) ( seed ) * 4.656612875E-10;

  value = a + ( b - a ) * value;

  return value;
}

/*
 * report: reports progress of the simulation
 *
 * Local: avg, the average population fitness
 * Local: double square_sum, square of sum for std calc
 * Local: double stddev, standard deviation of population fitness
 * Local: double sum, the total population fitness
 * Local: double sum_square, sum of squares for std calc
 */
void report ( int generation ){

  double avg;
  double best_val;
  int i;
  double square_sum;
  double stddev;
  double sum;
  double sum_square;

  if ( generation == 0 ){
    cout << "\n";
    cout << "  Generation       Best            Average       Standard \n";
    cout << "  number           value           fitness       deviation \n";
    cout << "\n";
  }

  sum = 0.0;
  sum_square = 0.0;

  for ( i = 0; i < POPSIZE; i++ ){
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


/*
 * selector: is the selection function
 *
 * (Standard proportional selection for maximization problems incorporating the elitist model.  This makes sure that the best member always survives)
 *
 *  Input/output: &SEED, a seed for the random number generator
 */
void selector ( int &seed ){

  const double a = 0.0;
  const double b = 1.0;
  int i;
  int j;
  int mem;
  double p;
  double sum;

  //  Find the total fitness of the population.
  sum = 0.0;
  for ( mem = 0; mem < POPSIZE; mem++ ){
    sum = sum + population[mem].fitness;
  }

  //  Calculate the relative fitness of each member.
  for ( mem = 0; mem < POPSIZE; mem++ ){
    population[mem].rfitness = population[mem].fitness / sum;
  }

  //  Calculate the cumulative fitness.
  population[0].cfitness = population[0].rfitness;
  for ( mem = 1; mem < POPSIZE; mem++ ){
    population[mem].cfitness = population[mem-1].cfitness +
      population[mem].rfitness;
  }

  //  Select survivors using cumulative fitness.
  for ( i = 0; i < POPSIZE; i++ ){
    p = r8_uniform_ab ( a, b, seed );
    if ( p < population[0].cfitness ){
      newpopulation[i] = population[0];
    }
    else{
      for ( j = 0; j < POPSIZE; j++ ){
        if ( population[j].cfitness <= p && p < population[j+1].cfitness ){
          newpopulation[i] = population[j+1];
        }
      }
    }
  }

  //  Overwrite the old population with the new one.
  for ( i = 0; i < POPSIZE; i++ ){
    population[i] = newpopulation[i];
  }

  return;
}


/*
 * Xover: performs crossover of the two selected parents
 *
 * Local: point, the crossover point
 * Input: ONE, TWO, the indices of the two parents
 * Input/output: &SEED, a seed for the random number generator
 */
void Xover ( int one, int two, int &seed ){

  int i;
  int point;
  double t;

  //  Select the crossover point.
  point = i4_uniform_ab ( 0, NVARS - 1, seed );

  //  Swap genes in positions 0 through POINT-1.
  for ( i = 0; i < point; i++ )
  {
    t                       = population[one].gene[i];
    population[one].gene[i] = population[two].gene[i];
    population[two].gene[i] = t;
  }

  return;
}


/*
 * Función Ackley usando CUDA
 *
 */
__global__ void
gaAckleyCUDA(float *C, float *A, int wA, int hA, float valorA, float valorB, float valorC)
{

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;
    float tmpSum2 = 0;
    float term1 =0, term2=0;

    if (ROW < wA) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < hA; i++) {
            tmpSum += A[i * hA + COL] * A[i * hA + COL];
            tmpSum2 += cos(valorC*A[i * hA + COL]);
        }
    }

    term1= (0-valorA)*exp( (0-valorB)*sqrt(tmpSum/wA) );
    term2= 0-exp(tmpSum2/wA);

    C[COL * hA + ROW] = term1 + term2 + valorA + exp(1.0);

}


/*
 * Función Rastrigin usando CUDA
 *
 */
__global__ void
gaRastriginCUDA(float *C, float *A, int wA, int hA, float valorAR)
{

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < wA) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < hA; i++) {
            tmpSum += (A[i * hA + COL]*A[i * hA + COL]) - (valorAR * cos(2 * M_PI * A[i * hA + COL]) );
        }
    }

    C[COL * hA + ROW] = (valorAR * wA) + tmpSum;


}


/*
 * constantInit
 */
void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}


/*
 * init
 *
 * Después de usar la función initialize()  para asignar valores a la población con las restricciones necesarias, copiamos esos valores en la matriz que usará el problema
 */
void init(float *data){

    int member;
    int i,j=0;
    double x[POPSIZE * NVARS];
    //printf("\n");
    for ( i = 0; i < NVARS; i++ ){
    	for ( member = 0; member < POPSIZE; member++ ){
			x[i] = population[member].gene[i];
			//printf("   %f",x[i]);
			data[j] = x[i];
			j++;
      }
    }

}


/*
 * resultToHost
 *
 * Después de evaluar la matriz correspondiente, almacenamos los resultados
 */
void resultToHost(float *data, int size){
    int member;
    //printf("\n");
    for ( member = 0; member < POPSIZE; member++ ){
    	population[member].fitness = data[member*NVARS];
    	//printf("   %f",population[member].fitness);
    }
}


/**
 * Run using CUDA
 */
int geneticAlgorithm(int argc, char **argv, dim3 &dimsA, int ag_rastrigin, float valorARastrigin, int max_gen, float min, float max, int n_vars, float p_mutation, int population_size, float p_crossover, float valorA, float valorB, float valorC)
{

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

	POPSIZE = population_size;
	MAXGENS = max_gen;
	NVARS = n_vars;
	MIN = min;
	MAX = max;
	PXOVER = p_crossover;
	PMUTATION = p_mutation;



    // Allocate host memory for matrices A
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    // Initialize host memory
    //constantInit(h_A, size_A, 1);

	int generation;
	int i;
	int seed;


	seed = 123456789;
	initialize ( seed );
	//
	init(h_A);


    // Allocate device memory
    float *d_A,  *d_C;

	// Allocate host vector C
    unsigned int mem_size_C = dimsA.x * dimsA.y * sizeof(float);
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
    dim3 threads(dimsA.x, dimsA.y);
    dim3 grid(1, 1);
/*	if (dimsA.x*dimsA.y > 512){
		threads.x = 512;
		threads.y = 512;
		grid.x = ceil(double(dimsA.x)/double(threads.x));
		grid.y = ceil(double(dimsA.y)/double(threads.y));
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

	if(ag_rastrigin==1){

		gaRastriginCUDA<<< grid, threads >>>(d_C, d_A, dimsA.x, dimsA.y, valorARastrigin);
	    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	    resultToHost(h_C, dimsA.x * dimsA.y);

	    keep_the_best ( );

		for ( generation = 0; generation < MAXGENS; generation++ ){
		  selector ( seed );
		  crossover ( seed );
		  mutate ( seed );
		  //report ( generation );
		  gaRastriginCUDA<<< grid, threads >>>(d_C, d_A, dimsA.x, dimsA.y, valorARastrigin);
		  cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
		  resultToHost(h_C, dimsA.x * dimsA.y);

		  elitist ( );
		}

	}
	else{

	    gaAckleyCUDA<<< grid, threads >>>(d_C, d_A, dimsA.x, dimsA.y, valorA, valorB, valorC);
	    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	    resultToHost(h_C, dimsA.x * dimsA.y);

	    keep_the_best ();

		for ( generation = 0; generation < MAXGENS; generation++ ){
		  selector ( seed );
		  crossover ( seed );
		  mutate ( seed );
		  //report ( generation );
		  gaAckleyCUDA<<< grid, threads >>>(d_C, d_A, dimsA.x, dimsA.y, valorA, valorB, valorC);
		  cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
		  resultToHost(h_C, dimsA.x * dimsA.y);

		  //keep_the_best ( );
		  elitist ( );
		}


	}


	printf(" \"resultado_AG\" : {\n");

	printf("   \"generations\" : \"%d\", \n", MAXGENS);

	printf("   \"values\" : {\n");

	for ( i = 0; i < NVARS-1; i++ )
	{
     printf("      \"%d\": \"%f\", \n", i, population[POPSIZE].gene[i]);
	}
	 printf("      \"%d\": \"%f\" \n", NVARS-1, population[POPSIZE].gene[NVARS-1]);
	printf("   }, \n ");
	printf("  \"best_fitness\" : \"%f\" \n", population[POPSIZE].fitness);

	//  Terminate simple GA

	printf(" }, \n ");


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
    float msec = msecTotal ;
    double flops = (double)dimsA.x * (double)dimsA.y ;
    double gigaFlops = (flops) / (msec / 1000.0f);
    printf(
        " \"datos_computo\" : { \n   \"performance\" : \"%.2f Flop/s\", \n   \"time\" : \"%.3f msec\", \n   \"size\" : \"%.0f ps\", \n   \"workgroupSize\" : \"%u threads/block\" \n } \n",
        gigaFlops,
        msec,
        flops,
        threads.x);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }


/*
    printf(" \"resultados (solo de la ULTIMA generacion)\" : { \n");


    for (int i = 0; i < (int)(dimsA.x * dimsA.y); i++){
    	if(i%dimsA.y==0){
    		printf("   \"fitness population %d\" : \"%.8f\", \n", i/dimsA.y, h_C[i]);
    	}
    }

    printf("   \"x\": \"x\" \n }");
*/

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

        printf("Values to the genetic algorithm: \n");
        printf("      -max_gen= maximum number of generations (100 by default)\n");
        printf("      -min= minimum individual value \n");
        printf("      -max= maximum individual value \n");
        printf("      -p_mutation= probability of mutation (0.15 by default)\n");
        printf("      -population_size= population size (50 by default)\n");
        printf("      -p_crossover= probability of crossover (0.8 by default)\n");
        printf("      -n_vars= number of problem variables  (10 by default)\n");

        printf("Values to Ackley optimization Test Function: \n");
        printf("      -a= (20 by default) -b= (0.2 by default) -c= (2*PI by default) \n");
        printf("       and min=-32.768 and max=32.768 by default \n");

        printf("Values to Rastrigin optimization Test Function: \n");
        printf("      -A_R= (10 by default) \n");
        printf("       and min=-5.12 and max=5.12 by default \n");

        printf("\nExecute Ackley by default, -rastrigin=1 to execute rastrigin function \n\n");

        exit(EXIT_SUCCESS);
    }


	// Vemos que algoritmo vamos a aplicar (Ackley o Rastrigin)
	int ag_rastrigin = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "rastrigin"))
    {
    	ag_rastrigin = getCmdLineArgumentInt(argc, (const char **)argv, "rastrigin");

    	if(ag_rastrigin > 1 or ag_rastrigin < 0){
    		printf("INPUT ERROR: Value to -rastrigin= should be 0 or 1 \n");
    		exit(EXIT_SUCCESS);
    	}
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device")){
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
    /*else
    {
        printf(" \"dispositivo\" : \"GPU Device %d: '%s' with compute capability %d.%d\", \n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }*/


    // Matrix dimensions (width x height)
    dim3 dimsA(POPSIZE, NVARS, 1);

    // Variables para el algoritmo rastrigin
    float valorARastrigin=10;
    if (checkCmdLineFlag(argc, (const char **)argv, "A_R"))
    {
    	valorARastrigin = getCmdLineArgumentFloat(argc, (const char **)argv, "A_R");
    }

    // Variables para el algoritmo Ackley
    float valorA=20;
    if (checkCmdLineFlag(argc, (const char **)argv, "a"))
    {
    	valorA = getCmdLineArgumentFloat(argc, (const char **)argv, "a");
    }

    float valorB=0.2;
    if (checkCmdLineFlag(argc, (const char **)argv, "b"))
    {
        valorB = getCmdLineArgumentFloat(argc, (const char **)argv, "b");
    }

    float valorC=2*M_PI;
    if (checkCmdLineFlag(argc, (const char **)argv, "c"))
    {
        valorC = getCmdLineArgumentFloat(argc, (const char **)argv, "c");
    }


    // Variables para el Algoritmo genético

    int max_gen=100;
    if (checkCmdLineFlag(argc, (const char **)argv, "max_gen")){
    	max_gen = getCmdLineArgumentInt(argc, (const char **)argv, "max_gen");
    }
    if(max_gen < 0){
        printf("INPUT ERROR: Value to -max_gen= should be positive \n");
        exit(EXIT_SUCCESS);
    }

    float min=-32.768;
    float max=32.768;

    if(ag_rastrigin==1){
        min=-5.12;
        max=5.12;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "min")){
    	min = getCmdLineArgumentFloat(argc, (const char **)argv, "min");
    }


    if (checkCmdLineFlag(argc, (const char **)argv, "max")){
    	max = getCmdLineArgumentFloat(argc, (const char **)argv, "max");

    }
    if(max < min){
        printf("INPUT ERROR: Maximum value should be greater than the minimum  \n");
        exit(EXIT_SUCCESS);
    }

    //    NVARS is the number of problem variables.
    int n_vars=10;
    if (checkCmdLineFlag(argc, (const char **)argv, "n_vars")){
    	dimsA.y = n_vars = getCmdLineArgumentInt(argc, (const char **)argv, "n_vars");
    }
    if(n_vars >= MAXVARS){
        printf("INPUT ERROR: Value to -n_vars= should be less than %d \n", MAXVARS);
        exit(EXIT_SUCCESS);
    }
    float p_mutation=0.15;
    if (checkCmdLineFlag(argc, (const char **)argv, "p_mutation")){
    	p_mutation = getCmdLineArgumentFloat(argc, (const char **)argv, "p_mutation");
    }
    if(p_mutation<0 or p_mutation>1){
        printf("INPUT ERROR: Value to -p_mutation= should be between 0 and 1 \n");
        exit(EXIT_SUCCESS);
    }

    int population_size=50;
    if (checkCmdLineFlag(argc, (const char **)argv, "population_size")){
    	dimsA.x = population_size = getCmdLineArgumentInt(argc, (const char **)argv, "population_size");
    }
    if(population_size<0 or population_size>=MAXPOPSIZE){
        printf("INPUT ERROR: Value to -p_crossover= should be less than %d \n", MAXPOPSIZE);
        exit(EXIT_SUCCESS);
    }

    float p_crossover=0.8;
    if (checkCmdLineFlag(argc, (const char **)argv, "p_crossover")){
    	p_crossover = getCmdLineArgumentFloat(argc, (const char **)argv, "p_crossover");
    }
    if(p_crossover<0 or p_crossover>1){
        printf("INPUT ERROR: Value to -p_crossover= should be between 0 and 1 \n");
        exit(EXIT_SUCCESS);
    }

	// Salida del programa en formato JSON
	printf("{ \"calculo\":{ \n");

	if(ag_rastrigin==1){
		//printf(" \"nombre\" : \"Rastrigin function in genetic algorithm using CUDA\", \n");
		printf(" \"nombre\" : \"Algoritmo genético usando la función de Rastrigin mediante CUDA\", \n");
	}
	else{
		//printf(" \"nombre\" : \"Ackley function in genetic algorithm using CUDA\", \n");
		printf(" \"nombre\" : \"Algoritmo genético usando la función de Ackley mediante CUDA\", \n");
	}

    //printf(" \"dispositivo\" : \"GPU Device %d: '%s' with compute capability %d.%d\", \n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    printf(" \"dispositivo\" : \"GPU Device %d: %s \", \n", devID, deviceProp.name);
    printf(" \"capacidad_computo\" : \" %d.%d \", \n",  deviceProp.major, deviceProp.minor);

    printf(" \"info_matriz\" : \"Matrix(%d,%d) with random values\", \n", dimsA.x, dimsA.y);

	if(ag_rastrigin==1){
	    printf(" \"info_input\" : { \n   \"A\" : \"%f\", \n   \"n_generations\" : \"%d\",\n   \"minimal_value\" : \"%.4f\",\n   \"maximum_value\" : \"%.4f\",\n   \"p_mutation\" : \"%f\",\n   \"p_crossover\" : \"%f\",\n   \"population_size\" : \"%d\",\n   \"n_vars\" : \"%d\" \n }, \n", valorARastrigin, max_gen, min, max, p_mutation, p_crossover, population_size, n_vars);
	}
	else{
	    printf(" \"info_input\" : { \n   \"a\" : \"%f\", \n   \"b\" : \"%f\", \n   \"c\" : \"%f\", \n   \"n_generations\" : \"%d\",\n   \"minimal_value\" : \"%.4f\",\n   \"maximum_value\" : \"%.4f\",\n   \"p_mutation\" : \"%f\",\n   \"p_crossover\" : \"%f\",\n   \"population_size\" : \"%d\",\n   \"n_vars\" : \"%d\" \n }, \n", valorA, valorB, valorC, max_gen, min, max, p_mutation, p_crossover, population_size, n_vars);
	}

    int matrix_result = geneticAlgorithm(argc, argv, dimsA, ag_rastrigin, valorARastrigin, max_gen, min, max, n_vars, p_mutation, population_size, p_crossover, valorA, valorB, valorC);

    printf("} \n}");

    exit(matrix_result);

}
