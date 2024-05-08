import keras
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
#from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import EarlyStopping
from keras import layers
from keras.layers import Dropout
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import PIL
from PIL import Image
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns
import warnings

tumor_images = 'data/Tumor'
healthy_images = 'data/Healthy'
dataset = 'data'

tumor_data = [i for i in os.listdir(tumor_images)]
healthy_data = [i for i in os.listdir(healthy_images)]

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
for i in range(6):
	image_path = os.path.join(tumor_images, tumor_data[i])
	img = mpimg.imread(image_path)
	ax = axes[i // 3, i % 3]
	ax.imshow(img)
	ax.axis('off')
#plt.show()

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
for i in range(6):
	image_path = os.path.join(healthy_images, healthy_data[i])
	img = mpimg.imread(image_path)
	ax = axes[i // 3, i % 3]
	ax.imshow(img)
	ax.axis('off')
#plt.show()

img_height, img_width = 256, 256
batch_size = 200 #the subset of imagesgoing through the training section of 
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # normalize pixel values
    validation_split=0.2,  # reserve some images for validation
)
training_data = datagen.flow_from_directory( #there are 3681 images belonging to the two classes of Healthy and Tumor data, and 919 of them are part of the validation dataset 
	'data', #dataset,
	target_size=(img_height, img_width),
	batch_size=batch_size,
	class_mode='binary', #tumor or no-tumor
	color_mode='grayscale',
	interpolation='bilinear', #how the model will introduce randomness to the images (resize or change the image)
	subset='training') #validation 
validation_data = datagen.flow_from_directory(
	'data', #dataset,
	target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
    interpolation='bilinear', # strategy used for resampling when resizing
    subset='validation')

metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.AUC(name='prc', curve='PR')]


model = Sequential()
model.add(layers.Conv2D(8, (3, 3), activation="relu", padding="same", input_shape=(img_height, img_width, 1)))
model.add(layers.MaxPooling2D(2,2)) #
model.add(layers.Conv2D(8, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(8, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(8, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.8))
model.add(layers.Dense(1, activation="sigmoid"))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)
csv_logger = CSVLogger('training_log.csv', append=True, separator=',')
epoch = 10 #how many times model goes through training images 
history = model.fit(training_data, validation_data=validation_data, epochs=epoch, verbose=1, callbacks=[es, csv_logger])
val_loss, val_accuracy, val_recall, val_precision, val_auc = model.evaluate(validation_data)
model.save('model.h5')
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Recall: {val_recall}")
print(f"Validation Precision: {val_precision}")
print(f"Validation AUC (PR Curve): {val_auc}")


