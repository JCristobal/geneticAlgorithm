# matrizAckley

Algoritmo genético, optimizado con la función Ackley y Rastrigin, de forma paralela usando **CUDA 7.5**

Ejcutable [*geneticAlgorithm*](https://github.com/JCristobal/geneticAlgorithm/blob/master/geneticAlgorithm) con parámetros: 

* ? para desplegar información
* -device= para elegir el disposivo donde computar
* -cvalue= valor constante de las filas (por defecto 1)
* -max_gen= número de generaciones (100 por defecto)
* -min= valor mínimo individual ( -32.768 por defecto para Ackley y -5.12 en Rastrigin)
* -max= valor máximo individual ( 32.768 por defecto para Ackley y 5.12 en Rastrigin)
* -p_mutation= probabilidad de mutación (0.15 por defecto)
* -population_size= tamaño de la población (50 por defecto)
* -p_crossover= probabilidad de cruce (crossover) (0.8 por defecto)
* -n_vars= número de variables de cada población  (10 por defecto)
* -a= valor para el parámetro *a* propio de la función Ackley (20 por defecto) 
* -b= valor para el parámetro *b* propio de la función Ackley (0.2 por defecto) 
* -c= valor para el parámetro *c* propio de la función Ackley (2*PI por defecto)
* -A_R= valor para el parámetro *A* propio de la función Rastrigin


Por defecto se evaluará con la función Ackley, para evaluar mediante Rastrigin añadimos el parámetro **-rastrigin=1**

La salida se dará en formato JSON, dando información del cómputo, junto a los resultados según los parámetros dados.

Dicho ejecutable se podra utilizar en un [servicio web](https://github.com/JCristobal/SWGPU) que ofrecerá una computación paralela usando **CUDA 7.5** en un servidor externo.



