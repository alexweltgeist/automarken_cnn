# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:08:26 2021

@author: alex
"""

# Install if missing
# ! pip install tensorflow_hub

import os, shutil
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Download data from Kaggle or Git to local directory ('/test')
# https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

# subfolder erstellen
sourcefile = 'C:/Users/alex/CAS_ML_local/B_Deeplearning/03_Project/test'

'''
# Filename auslesen und Orderstruktur anlegen
for filename in os.listdir(sourcefile): 
    brand = filename.rsplit('_', 17)[0]
    try:
        os.mkdir(os.path.join(sourcefile, brand))   # Ordner erstellen wenn nötig...
    except WindowsError:
        pass                                        # ...sonst weiter und Bild moven
    shutil.move(os.path.join(sourcefile, filename), os.path.join(sourcefile, brand, filename))

# Anzahl Bilder zählen
def get_nr_files(sourcefile):
    file_count = 0
    for r, d, files in os.walk(sourcefile):
        file_count += len(files)
    return file_count
# shoud return 64467
'''

# Daten in TensorFlow laden und spliten
para_kwargs = dict(
    directory=sourcefile, 
    labels='inferred', 
    label_mode='categorical',   # for fitting = 'categorical' for print = 'int'
    class_names=None, 
    color_mode='rgb', 
    batch_size=32, 
    image_size=(224, 224), 
    shuffle=False, 
    seed=None, 
    validation_split=0.2,
    interpolation='bilinear', 
    follow_links=False)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    **para_kwargs, 
    subset='training')

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    **para_kwargs, 
    subset='validation')

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)

# Visualisieren der eingelesenen Daten
class_names = train_ds.class_names

image_batch, label_batch = next(iter(train_ds))
image_batch / 255

plt.figure(figsize=(10, 10))
for i in range(16):
  ax = plt.subplot(4, 4, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")


# Pre-trained Model auswählen (aus TF-Hub)
model_name = "mobilenet_v2_100_224" 
model_handle_map = {"mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",}
model_handle = model_handle_map.get(model_name)
IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 32
print(f"\nAusgewaehltes model: {model_name} : {model_handle}")
print(f"\nBild-Groesse {IMAGE_SIZE}")

# CNN zusammenstellen mit ANzahl Klassen wie im Datenset 
print("\nModell erstellen mit", model_handle)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE)),
    hub.KerasLayer(model_handle, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(class_names), 
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE)
model.summary()

# Trainieren des Modells 
model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=25, 
    validation_data=valid_ds).history

# plot the development of the accuracy and loss during training
plt.figure(figsize=(12,4))
plt.subplot(1,2,(1))
plt.plot(history['accuracy'],linestyle='-.')
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.subplot(1,2,(2))
plt.plot(history['loss'],linestyle='-.')
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


pred=model.predict(valid_ds)

'''
Wie machen wir Test datenset?
Wie zeigen wir confusionmatrix ?
Wie zeigen wir predicted foto?

Cool wäre:
    eigenes modell
    alles auf colab migrieren

print(confusion_matrix(np.argmax(valid_ds.image_batch,axis=1),np.argmax(pred,axis=1)))
acc_fc = np.sum(np.argmax(valid_ds,axis=1)==np.argmax(pred,axis=1))/len(pred)
print("Acc = " , acc_fc)

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n]==1][0].title())
        plt.axis('off')

show_batch(train_ds, train_ds)
'''


