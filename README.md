## Librerias necesarias para correr el proyecto

Es importante entender que este proyecto se debe correr sobre un venv, para garatizar que no hayan daños
en los demas entornos de python.

Se hace uso de las librerias pandas, numpy, matplotlib y skopt (scikit-optimize). A continuación se colocan los comandos para hacer
la instalación necesaria sobre el venv:

`pip install pandas scikit-optimize matplotlib ta-lib polars`
## Cómo funciona?

El funcionamiento es bastante simple. Primero debemos fijar qué objetivo tenemos, ya sea, tener buenos indicadores técnicos 
o un buen rendimiento monetario. Para realizar eso, se usa el archivo find_best, en el que se le puede pasar el nombre del archivo
o el DataFrame con la información, y solo es esperar a que el optimizador termine el proceso.

Este proyecto garantiza buenos indicadores pero no los mejores.


## Cómo maximizamos?
Se usan los métodos de optimización gaussiana o bayesiana, y minimización con arboles, ambos métodos de la libreria de skopt.


En el archivo `find_best.py` se pueden encontrar dos funciones principales, `best_main()` y `best_partition()`. 
La primera toma todos los datos proporcionados y encuentra la mejor moving averague optimizando los kpis o la cantidad de dinero ganada. 

La segunda siempre optimiza los kpis, pero con la diferencia que se hace una partición de la información; se le pasa un 
número y con respecto a ese número parte la base de datos para obtener una parte de entrenamiento y tomar toda la info de testeo. Esto mas como
un experimento de qué tanto se mantiene la mejor moving averague encontrada en la totalidad del tiempo.

