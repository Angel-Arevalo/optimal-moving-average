## Librerias necesarias para correr el proyecto

Es importante entender que este proyecto se debe correr sobre un venv, para garatizar que no hayan daños
en los demas entornos de python.

Se hace uso de las librerias pandas, numpy y skopt (scikit-optimize). A continuación se colocan los comandos para hacer
la instalación necesaria sobre el venv:

`pip install pandas`
`pip install scikit-optimize`


## Cómo funciona?

El funcionamiento es bastante simple. Primero debemos fijar qué objetivo tenemos, ya sea, tener buenos indicadores técnicos 
o un buen rendimiento monetario. Para realizar eso, se usa el archivo find_best, en el que se le puede pasar el nombre del archivo
o el DataFrame con la información, y solo es esperar a que el optimizador termine el proceso.

Este proyecto garantiza buenos indicadores pero no los mejores.
