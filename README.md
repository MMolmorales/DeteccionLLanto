# DeteccionLLanto
Se crea modelo de deep learning para la detección de llanto de un bebe, enmarcado en el proyecto personal de monitero de neonatos, este modelo es uno de los dos modelos que se realizaran para ayudar a los cuidadores de recién nacidos. 
Para este modelo se utiliza el dataset de donate-a-cry junto al dataset de kaggle de sonidos generales, por lo que las clases de salida son 'babycry' y 'others'.
Debido a que se trabajan con archivos de sonido se utiliza librosa para la creación de la representación en imagenes, luego se utilza tensorflow con keras para la creación de una red neuronal CNN.
Se adjuntan las metricas de evaluación correspondientes para esta modelo en los archivos con nombre métrica. 

<table>
  <tr>
    <td>
        <img src="example_s.jpg" alt="Imagen 1" width="500"><br>
        <p>   Espectrograma del llanto de un bebe  </p>
    </td>
    <td>
        <img src="example_m.jpg" alt="Imagen 2" width="500"><br>
        <p>   Gráfico de los coeficientes mfcc para el llanto de un bebe  </p>
    </td>
  </tr>
</table>

Se presentan las metricas de evaluación para una red CNN utilizando los espectogramas de cada archivo de audio, en el proceso de entrenamiento se utiliza un learning rate adapativo según las metricas de entrenamiento. 
Observar que la precisión por epoca aumenta por cada una de las epocas, para los datos de entrenamiento y validación, alcanzando valores entorno al 90% de precisión para las 50 epocas de entrenamiento. Además, con los datos de test
se realiza la matriz de confusión obteniendo una precisión del 91%.


<table>
  <tr>
    <td>
      <img src="Costo_Epoca_S.png" alt="Imagen 1" width="500">
    </td>
    <td>
      <img src="Precision_Epoca_S.png" alt="Imagen 2" width="500">
    </td>
    <td>
      <img src="MC S.png" alt="Imagen 3" width="500">
    </td>
  </tr>
    <tr>
    <td colspan="3">
      <center><p>          Metricas de evaluación para la red CNN de los espectrogramas.     </p></center>
    </td>
  </tr>
</table>
