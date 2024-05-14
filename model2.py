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
model = keras.models.load_model('model.h5')
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Recall: {val_recall}")
print(f"Validation Precision: {val_precision}")
print(f"Validation AUC (PR Curve): {val_auc}")

predictions = []
actualvals = []

for _ in range(66):
    test, actual = vaildation_data.next()
    
    
    actualvals.extend(actual)

    prediction = model.predict(test)
    

    predictions.extend(prediction.flatten() > 0.5)

# Calculate confusion matrix outside the loop
cmatrix = confusion_matrix(actualvals, predictions)


# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Actual Classification')
plt.ylabel('Predicted Classification')
plt.title('Confusion Matrix')
#plt.show()



# ROC Curve and AUC
y_true = np.array(actualvals)
y_pred = np.array(predictions).astype(float)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
#plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
#plt.show()


# Sample Predictions
def display_sample_predictions(model, data_generator, num_images=6):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(num_images):
        img, label = data_generator.next()
        prediction = model.predict(img)
        ax[i // 3, i % 3].imshow(img[0], cmap='gray')
        ax[i // 3, i % 3].set_title(f"True: {label[0]}, Pred: {int(prediction[0] > 0.5)}")
        ax[i // 3, i % 3].axis('off')
    plt.show()

display_sample_predictions(model, vaildation_data)


fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharex=True)
axs[0].plot(range(1, epoch+1), history.history['accuracy'], label='Training Accuracy')
axs[0].plot(range(1, epoch+1), history.history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch') 
axs[0].set_ylabel('Accuracy')
axs[0].set_xticks(np.arange(1, epoch+1, 1))
axs[0].legend()

axs[1].plot(range(1, epoch+1), history.history['loss'], label='Training Loss')
axs[1].plot(range(1, epoch+1), history.history['val_loss'], label='Validation Loss')
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('Epoch') 
axs[1].set_ylabel('Loss')
axs[1].set_xticks(np.arange(1, epoch+1, 1))
axs[1].legend()

plt.show()


