# DeteccionLLanto
Se crea modelo de deep learning para la detección de llanto de un bebe, enmarcado en el proyecto personal de monitero de neonatos, este modelo es uno de los dos modelos que se realizaran para ayudar a los cuidadores de recién nacidos. 
Para este modelo se utiliza el dataset de donate-a-cry junto al dataset de kaggle de sonidos generales, por lo que las clases de salida son 'babycry' y 'others'.
Debido a que se trabajan con archivos de sonido se utiliza librosa para la creación de la representación en imagenes, luego se utilza tensorflow con keras para la creación de una red neuronal CNN.
Se adjuntan las metricas de evaluación correspondientes para esta modelo en los archivos con nombre métrica. 

