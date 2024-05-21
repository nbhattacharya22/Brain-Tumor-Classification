import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

########################################### Create the UI ###########################################

model = tf.keras.models.load_model('model.h5') # Load the model that we desire to use

st.markdown("<h1 style='text-align: center;'>Brain Tumor Image Classification</h1>", unsafe_allow_html=True) # Title of the app

st.markdown("<h6 style='text-align: center;'> Developed by: Neel, Ehsan, Arnay, Alexandra, Neera, and Oscar! </h6>", unsafe_allow_html=True) # Subtitle of the app, containing names of the developers

if 'predicted_class' in st.session_state: #Initialize or clear previous prediction state 
    del st.session_state['predicted_class']
    
uploaded_image = st.file_uploader("Upload an X-ray Image:", type=["png", "jpg", "jpeg"]) # File uploader to upload the image, only png, jpg, and jpeg files are allowed
if uploaded_image is not None:  #if an image is uploaded, display the image
    img = load_img(uploaded_image, target_size=(256, 256), color_mode='grayscale')  # Load the image and convert it to grayscale
    st.image(img, caption="Uploaded X-ray Image", use_column_width=True) # Display the image on the app
    image_array = img_to_array(img)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

col1, col2, col3 = st.columns([1,0.3,1]) # Create columns for the buttons, this helps to center the button on the app
with col2:
    predict_button = st.button("Predict") # Button to predict the image

if predict_button:     
    if uploaded_image is not None: # If the user has uploaded an image, then predict the image
        prediction = model.predict(image_array) 
        predicted_class = "No Tumor is Detected" if prediction[0] > 0.5 else "A Tumor is Detected" #If the prediction is greater than 0.5, then the image is classified as having a tumor, otherwise it is classified as not having a tumor
        st.session_state['predicted_class'] = predicted_class # Store the prediction in the session state
    else: # If the user tries to predict without uploading an image, display an error message
        st.markdown("<h6 style='text-align: center;'> Please upload an image before predicting! </h6>", unsafe_allow_html=True) # Display an error message on the app

if 'predicted_class' in st.session_state: # If the prediction has been made, display the prediction
    st.markdown(f"<h4 style='text-align: center;'>Based off the X-ray image uploaded: <b>{st.session_state['predicted_class']}</b></h4>", unsafe_allow_html=True) # Display the prediction on the app


########################################### Graphical Analysis ###########################################

st.markdown("<h1 style='text-align: center;'>Graphical Analysis</h1>", unsafe_allow_html=True)

st.image("Accuracy_and_validation_loss.png", caption="Graph 1: graph displays the model's accuracy over the training epochs for both the training and validation datasets as well as the loss over epochs for both traning and valiation phases.", width = 500, use_column_width=True)
st.image("precision_recall_curve.png", caption="Graph 2: graph shows the trade-off between precision and recall for different thresholds. Precision quantifies the accuracy of positive predictions, while recall measures the model’s ability to identify all relevant instances.", width = 500, use_column_width=True)
st.image("Confusion_matrix.png", caption="Graph 3: represents the accuracy of the model's predictions. It shows the number of correct and incorrect predictions categorized by the model's actual and predicted classifications.", width = 500, use_column_width=True)
st.image("receiver_operating_characteristic.png", caption="Graph 4: plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings, evaluating the binary classifier's performance.", width = 500, use_column_width=True)
st.image("training_log.png", caption="Graph 5: graphs shows the models accuracy and precision as well as other useful metrics throughout each epoch", width = 500, use_column_width=True)