# -*- coding: utf-8 -*-
"""

@author: mirko
"""

# Carga de modulos de uso general y librosa para el trabajo con archivos .wav

import os 
import librosa 
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%% Se ajusta las rutas con las cuales se van a trabajar.

# Define el directorio de trabajo
os.chdir("D:/Magister/Aplicaciones con IA/DeteccionLLanto")
current_directory = os.getcwd()

# Definir rutas 
path_imagenes_s = "D:/Magister/Aplicaciones con IA/DeteccionLLanto/bebe_llanto_dataset_wav_imagenes_spectrum"
path_imagenes_m = "D:/Magister/Aplicaciones con IA/DeteccionLLanto/bebe_llanto_dataset_wav_imagenes_MFCC"

# Archivos .wav
Wav = "D:/Magister/Aplicaciones con IA/DeteccionLLanto/bebe_llanto_dataset_wav"
Clases = os.listdir(Wav)

# Crear las carpetas de destino si no existen
for path in [path_imagenes_s, path_imagenes_s]:
    for carpeta in Clases:
        path_Carpeta = os.path.join(path, carpeta)
        if not os.path.exists(path_Carpeta):
            os.makedirs(path_Carpeta)
            print(f'Carpeta creada: {path_Carpeta}')
        else:
                print(f'Carpeta ya existe: {path_Carpeta}')


#%% Creacion de los espectros y representacion MFCC

# Espectros

def obtener_espectrograma(archivo, path_imagenes_s):
    y, sr = librosa.load(archivo)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Crear la figura sin ejes
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')  # Eliminar los ejes
    
    # Crear el path completo para guardar la imagen
    nombre_archivo = os.path.basename(archivo).replace('.wav', '.jpg')
    path_completo = os.path.join(path_imagenes_s, nombre_archivo)
    
    # Guardar la imagen
    plt.savefig(path_completo, bbox_inches='tight', pad_inches=0, format='jpg')
    plt.close()

# MFCC o Mel-Frequency Cepstral Coefficients

def obtener_mfcc(archivo, path_imagenes_m):
    # Cargar el archivo de audio
    y, sr = librosa.load(archivo)
    
    # Calcular los MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Crear la figura sin ejes
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.axis('off')  # Eliminar los ejes
    
    # Crear el path completo para guardar la imagen
    nombre_archivo = os.path.basename(archivo).replace('.wav', '_mfcc.jpg')
    path_completo = os.path.join(path_imagenes_m, nombre_archivo)
    
    # Guardar la imagen
    plt.savefig(path_completo, bbox_inches='tight', pad_inches=0, format='jpg')
    plt.close()



# Aplicar la función a cada archivo en cada carpeta
for archivo in Clases:
    archivo_path = os.path.join(Wav, archivo )
    for root, dirs, files in os.walk(archivo_path):
        for file in files:
            file_path = os.path.join(root, file)
            espectro_path_s= os.path.join(path_imagenes_s,archivo)
            espectro_path_m= os.path.join(path_imagenes_m,archivo)
            obtener_espectrograma(file_path, espectro_path_s)
            obtener_mfcc(file_path, espectro_path_m)
            
#%% Creación de carpetas test

import random
import shutil

path_test_s = os.path.join(current_directory, 'test_s')
path_test_m = os.path.join(current_directory, 'test_m')



for path in [path_test_s, path_test_m]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Carpeta creada: {path}')
        for carpeta in Clases:
            carpeta_test_s = os.path.join(path, carpeta)
            if not os.path.exists(carpeta_test_s):
                os.makedirs(carpeta_test_s)
    else:
        print(f'La carpeta {path} ya existe.')




    
#%%
# Seleccionar y mover archivos
size_test_per_class = 50
for sof,path_test in [[path_imagenes_s, path_test_s],[path_imagenes_m, path_test_m]]:
    for clase in Clases:
        ruta_carpeta = os.path.join(sof, clase)
        archivos = os.listdir(ruta_carpeta)    
        seleccionados = random.sample(archivos, size_test_per_class)
    
        # Mover los archivos seleccionados a la carpeta 'test'
        for archivo in seleccionados:
            path_archivo = os.path.join(ruta_carpeta, archivo)
            path_test_clase = os.path.join(path_test, clase)
            path_destino = os.path.join(path_test_clase, archivo)
            shutil.move(path_archivo, path_destino)



#%% Librerias de tensorflow y keras

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Parámetros de carga de datos
batch_size = 32
img_height = 224
img_width = 224

# Carga las imágenes desde las carpetas
train_dataset_s = image_dataset_from_directory(
    path_imagenes_s,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

train_dataset_m = image_dataset_from_directory(
    path_imagenes_m,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

# División en conjunto de entrenamiento y validación
val_split = 0.4
train_size = int((1 - val_split) * len(train_dataset_s))
train_ds_s = train_dataset_s.take(train_size)
val_ds_s = train_dataset_s.skip(train_size)
train_ds_m = train_dataset_m.take(train_size)
val_ds_m = train_dataset_m.skip(train_size)

#%%  Red para espectro

# Data Augmentation

import tensorflow as tf


# Normalización de las imágenes y Data Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2)
])

# Aplicar augmentations y normalización
train_ds = train_ds_s.map(lambda x, y: (data_augmentation(x, training=True), y))
normalization_layer = Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_s.map(lambda x, y: (normalization_layer(x), y))

# Optimización del pipeline de datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#%% Creación del modelo 

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', 
                           input_shape=(img_height, img_width, 3),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compilar el modelo con un learning rate más bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Configurar callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=1e-7)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[learning_rate_reduction, early_stopping]
)


#%%  Guardado y carga del modelo 

# Gurdado
model.save("D:\Magister\Aplicaciones con IA\DeteccionLLanto/Deteccionllanto_s.h5")
#model.summary()
 
# Carga 
#model = tf.keras.models.load_model('D:\Magister\Aplicaciones con IA\DeteccionLLanto/Deteccionllanto_s.h5')

#%% Metricas
# Visualización del desempeño (costo y precisión)
plt.figure(figsize=(20, 15)) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Costo', fontsize = 34)
plt.xlabel('Epoca', fontsize = 34)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.legend(['Entrenamiento', 'Validación'], loc='upper right', fontsize = 30)
plt.grid(True)
plt.show()


plt.figure(figsize=(20, 15)) 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Precisión', fontsize = 34)
plt.xlabel('Epoca', fontsize = 34)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.legend(['Entrenamiento', 'Validación'], loc='lower right', fontsize = 30)
plt.grid(True)
plt.show()

#%% Realizo predicciones sobre los datos de test_s

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Crear un generador para las imágenes de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    path_test_s,
    target_size=(img_height, img_width),  
    batch_size=32,
    class_mode='categorical',  
    shuffle=False  # Importante para no mezclar las imágenes en la evaluación
)


loss, accuracy = model.evaluate(test_generator)
print(f'Pérdida: {loss}')
print(f'Precisión: {accuracy}')


predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)


class_labels = list(test_generator.class_indices.keys())

#%% Confusión matrix
true_classes = test_generator.classes
cm = confusion_matrix(true_classes, predicted_classes)

#%% Configurar matriz de confusión
plt.figure(figsize=(18, 14))  # Ajustar el tamaño de la figura
ESP = ['babycry', 'others']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ESP)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')

plt.title('Matriz de Confusión CNN Espectrograma', fontsize=40)  
plt.xlabel('Clase Predicha', fontsize=24)      
plt.ylabel('Clase Verdadera', fontsize=24)     
plt.xticks(fontsize=30)                        
plt.yticks(fontsize=30)                        
# Ajustar el tamaño de la fuente de los números dentro de la matriz
for text in disp.text_.ravel():
    text.set_fontsize(40)

plt.show()


#%% Ahora con MFCC



#%%  Red para espectro

# Data Augmentation

import tensorflow as tf


# Normalización de las imágenes y Data Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2)
])

# Aplicar augmentations y normalización
train_ds = train_ds_m.map(lambda x, y: (data_augmentation(x, training=True), y))
normalization_layer = Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_m.map(lambda x, y: (normalization_layer(x), y))

# Optimización del pipeline de datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#%% Creación del modelo 

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', 
                           input_shape=(img_height, img_width, 3),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compilar el modelo con un learning rate más bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Configurar callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=1e-7)

#early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
#    callbacks=[learning_rate_reduction, early_stopping]
    callbacks = [learning_rate_reduction]
)


#%%  Guardado y carga del modelo 

# Gurdado
model.save("D:\Magister\Aplicaciones con IA\DeteccionLLanto/Deteccionllanto_m.h5")
#model.summary()
 
# Carga 
#model = tf.keras.models.load_model('D:\Magister\Aplicaciones con IA\DeteccionLLanto/Deteccionllanto_s.h5')

#%% Metricas
# Visualización del desempeño (costo y precisión)
plt.figure(figsize=(20, 15)) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Costo', fontsize = 34)
plt.xlabel('Epoca', fontsize = 34)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.legend(['Entrenamiento', 'Validación'], loc='upper right', fontsize = 30)
plt.grid(True)
plt.show()


plt.figure(figsize=(20, 15)) 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Precisión', fontsize = 34)
plt.xlabel('Epoca', fontsize = 34)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.legend(['Entrenamiento', 'Validación'], loc='lower right', fontsize = 30)
plt.grid(True)
plt.show()

#%% Realizo predicciones sobre los datos de test_s

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Crear un generador para las imágenes de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    path_test_m,
    target_size=(img_height, img_width),  
    batch_size=32,
    class_mode='categorical',  
    shuffle=False  # Importante para no mezclar las imágenes en la evaluación
)


loss, accuracy = model.evaluate(test_generator)
print(f'Pérdida: {loss}')
print(f'Precisión: {accuracy}')


predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)


class_labels = list(test_generator.class_indices.keys())

#%% Confusión matrix
true_classes = test_generator.classes
cm = confusion_matrix(true_classes, predicted_classes)

#%% Configurar matriz de confusión
plt.figure(figsize=(18, 14))  # Ajustar el tamaño de la figura
ESP = ['babycry', 'others']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ESP)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')

plt.title('Matriz de Confusión CNN MFCC', fontsize=40)  
plt.xlabel('Clase Predicha', fontsize=24)      
plt.ylabel('Clase Verdadera', fontsize=24)     
plt.xticks(fontsize=30)                        
plt.yticks(fontsize=30)                        
# Ajustar el tamaño de la fuente de los números dentro de la matriz
for text in disp.text_.ravel():
    text.set_fontsize(40)

plt.show()
