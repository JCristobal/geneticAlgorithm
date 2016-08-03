# matrizAckley
Función Ackley a varios vectores, de forma paralela usando **CUDA 7.5**

Sumatoria de un vector dentro de una matriz, de forma paralela, usanda **CUDA 7.5**

Ejcutable [*matrixAckley*](https://github.com/JCristobal/matrizAckley/blob/master/matrixAckley) con parámetros: 

* ? para desplegar información
* -device= para elegir el disposivo donde computar
* -width= ancho de la matriz (por defecto 32)
* -height= largo de la matriz (por defecto 32)
* -cvalue= valor constante de las filas (por defecto 1)
* -a= valor para el parámetro *a* propio de la función Ackley (20 por defecto) 
* -b= valor para el parámetro *b* propio de la función Ackley (0.2 por defecto) 
* -c= valor para el parámetro *c* propio de la función Ackley (2*PI por defecto)

La salida se dará en formato JSON, dando información del cómputo, junto a los resultados según los parámetros dados.

Dicho ejecutable se podra utilizar en un [servicio web](https://github.com/JCristobal/SWGPU) que ofrecerá una computación paralela usando **CUDA 7.5** en un servidor externo.
