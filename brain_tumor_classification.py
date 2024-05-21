import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import PIL
from PIL import Image
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#################################################### visualize the data ####################################################

# create a variable for the path of the data set 
DataSet = "Brain_Tumor_Data_Set" #Path to the total data set
tumor_data = 'Brain_Tumor_Data_Set\Brain_Tumor' # Path to the tumor data
healthy_data = 'Brain_Tumor_Data_Set\Healthy' # Path to the healthy data

# use os to get the list of files in the directory for the two data sets
tumor_data_files = [i for i in os.listdir(tumor_data)]
healthy_data_files = [i for i in os.listdir(healthy_data)] 

#plotting the data 

#Tumor data images 
figs, axis = plt.subplots (2,3, figsize = (12, 20)) # create a 2x3 grid of subplots, this is for tumor data
for i in range(6): #loop throught the first 6 images in the tumor data set
    image_path = os.path.join(tumor_data, tumor_data_files[i]) # get the path of the image
    image = mpimg.imread(image_path)
    ax = axis[i//3, i%3] 
    ax.imshow(image)
    ax.axis('off') # turn off the axis, since we are doing images not plotting points 

#Healthy data images 
figs, axis = plt.subplots (2,3, figsize = (12, 20)) # create a 2x3 grid of subplots, this is for healthy data
for i in range(6): #loop throught the first 6 images in the healthy data set
    image_path = os.path.join(healthy_data, healthy_data_files[i]) # get the path of the image
    image = mpimg.imread(image_path)
    ax = axis[i//3, i%3] 
    ax.imshow(image)
    ax.axis('off') # turn off the axis, since we are doing images not plotting points 


#################################################### Processing the data ####################################################

#This will be done using tensorflow's ImageDataGenerator class
#We will be using a 80/20 split for the training and validation data 

img_height, img_width = 256, 256 #The height and width of the images, make them all the same size
batch_size = 256 #The maximum number of samples that will be generated in a batch

datagen = tf.keras.preprocessing.image.ImageDataGenerator ( #Create an instance of the ImageDataGenerator class
    rescale = 1./255, #Normalize pixel values 
    validation_split = 0.2 #20% of the data will be used for validation
)

training_data = datagen.flow_from_directory( #Create a generator for the training data
    DataSet,
    target_size = (img_height, img_width), #Resize the images to the specified height and width
    batch_size = batch_size, #The maximum number of samples that will be generated in a batch
    class_mode = 'binary', #Since we are doing binary classification, looking for 0s and 1s 
    color_mode = "grayscale", #Convert the images to grayscale
    interpolation = "bilinear", 
    subset = "training"  #80% of the data will be used for training
)

validation_data = datagen.flow_from_directory( #Create a generator for the validation data
    DataSet, 
    target_size = (img_height, img_width), #Resize the images to the specified height and width
    batch_size = batch_size, #The maximum number of samples that will be generated in a batch
    class_mode = 'binary', #Since we are doing binary classification, looking for 0s and 1s
    color_mode = "grayscale", #Convert the images to grayscale
    interpolation = "bilinear",     
    subset = "validation" #20% of the data will be used for validation
) 

#create a list of metrics that will be used to evaluate the model
metrics = [ tf.keras.metrics.BinaryAccuracy(name = "accuracy"), #The accuracy of the model, need high accuracy so that we can correctly classify the images
            tf.keras.metrics.Precision(name = "Precision"), #The precision of the model, need high precision so that we can find the tumors 
            tf.keras.metrics.Recall (name = "Recall"), #The recall of the model, need high recall so that we can catch all the tumors
            tf.keras.metrics.AUC (name = "prc", curve = "PR")]  #The area under the precision-recall curve


#################################################### Building the model ####################################################

#Create a sequential model
model = Sequential()

#creating the layers for the model
channels = 1 #The number of channels in the image, since we are converting the images to grayscale, there is only 1 channel if it was colors it would be 3
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (img_height, img_width, channels), padding = 'same'))

model.add(layers.MaxPooling2D(3,3)) #pooling layer 

model.add(layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same')) #another convolutional layer

model.add(layers.MaxPooling2D(3,3)) #pooling layer 

model.add(layers.Flatten()) #flatten the data

model.add(layers.Dense(64, activation = 'relu')) #dense layer

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.3)) #dropout layer

model.add(layers.Dense(1, activation = 'sigmoid')) #output layer

#optimizer, is this to prevent overfitting (aka when the model starts to lose accuracy after a while) 

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) 
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = metrics)
model.summary() #print the summary of the model 

es = EarlyStopping(monitor = "accuracy", mode = "min", verbose = 1, patience = 20)
csv_logger = CSVLogger("training_log.csv", append = True, separator = ",") #log the training data to a csv file
epoch = 10 
history = model.fit(training_data, validation_data = validation_data, epochs = epoch, verbose = 1, callbacks = [es, csv_logger]) #train the model
model.save("model11.h5") #save the model

#################################################### Evaluating the model ####################################################

val_loss, val_accuracy, val_recall, val_precision, val_auc = model.evaluate(validation_data) 
print(f"Validation Loss: {val_loss}") #print the validation loss
print(f"Validation Accuracy: {val_accuracy}") #print the validation accuracy
print(f"Validation Recall: {val_recall}") #print the validation recall
print(f"Validation Precision: {val_precision}") #print the validation precision
print(f"Validation AUC (PR Curve): {val_auc}") #print the validation AUC

#################################################### Visualizing the model ####################################################

predictions = []   # List to store all the predictions
actualvals = []   # List to store all the actual values

for _ in range(66): #66 is the number of batches in the validation data
    test, actual = next(validation_data)
    
    
    actualvals.extend(actual)

    prediction = model.predict(test)
    

    predictions.extend(prediction.flatten() > 0.5) #If the prediction is greater than 0.5, then it is a tumor, else it is healthy

# Calculate confusion matrix outside the loop
cmatrix = confusion_matrix(actualvals, predictions)

# ROC Curve and AUC
y_true = np.array(actualvals)
y_pred = np.array(predictions).astype(float)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
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
        img, label = next(data_generator)
        prediction = model.predict(img)
        ax[i // 3, i % 3].imshow(img[0], cmap='gray')
        ax[i // 3, i % 3].set_title(f"True: {label[0]}, Pred: {int(prediction[0] > 0.5)}")
        ax[i // 3, i % 3].axis('off')
    plt.show()

display_sample_predictions(model, validation_data)

# Plotting the confusion matrix
fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharex=True)
axs[0].plot(range(1, epoch+1), history.history['accuracy'], label='Training Accuracy')
axs[0].plot(range(1, epoch+1), history.history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch') 
axs[0].set_ylabel('Accuracy')
axs[0].set_xticks(np.arange(1, epoch+1, 1))
axs[0].legend()

# Plotting the loss
axs[1].plot(range(1, epoch+1), history.history['loss'], label='Training Loss')
axs[1].plot(range(1, epoch+1), history.history['val_loss'], label='Validation Loss')
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('Epoch') 
axs[1].set_ylabel('Loss')
axs[1].set_xticks(np.arange(1, epoch+1, 1))
axs[1].legend()

plt.show()
