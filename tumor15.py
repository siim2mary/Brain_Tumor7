import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import time

# Define image size
img_height, img_width = 224, 224

# Load and preprocess images
def load_images_and_labels(extract_dir):
    labels_dict = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    data = []
    labels = []

    for folder in os.listdir(extract_dir):
        folder_path = os.path.join(extract_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img = img_to_array(img)
                data.append(img)
                labels.append(labels_dict[folder])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

# Load model
model = load_model('CNN_image_classification_model.h5')

# Define the class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit app title and description
st.title("Brain Tumor Detection with CNN")
st.write("Upload an MRI image to classify it as Glioma, Meningioma, No Tumor, or Pituitary.")

# Sidebar for file upload and options
uploaded_file = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
evaluate = st.sidebar.checkbox("Evaluate on Validation/Test Data")
plot_history = st.sidebar.checkbox("Plot Training History")
save_model = st.sidebar.checkbox("Save Model")

# Initialize progress bar
progress_bar = st.sidebar.progress(0)

# Load test data and labels
base_dir = r'C:\Users\Joby\PycharmProjects\pythonProject1Brain_Tumor'
test_data_dir = os.path.join(base_dir, 'Testing')
test_data, test_labels = load_images_and_labels(test_data_dir)
test_data = test_data / 255.0  # Normalize test data
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=4)  # Convert test labels to categorical

# Display the uploaded image and make prediction
if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(img_height, img_width))
    st.image(img, caption='Uploaded MRI Image.', use_column_width=True)
    st.write("")

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    st.write(f"Predicted Class: **{predicted_class_name}**")

    # Identify the correct class for the uploaded image
    correct_class_name = None
    for folder in os.listdir(test_data_dir):
        folder_path = os.path.join(test_data_dir, folder)
        if os.path.isfile(os.path.join(folder_path, uploaded_file.name)):
            correct_class_name = folder
            break

    if correct_class_name:
        if predicted_class_name.lower() == correct_class_name.lower():
            st.success(f'The uploaded image was predicted correctly as {predicted_class_name}.')
        else:
            st.error(
                f'The uploaded image was predicted as {predicted_class_name}, but the correct class is {correct_class_name}.')
    else:
        st.error("Uploaded image does not match any class in the test dataset.")

# Evaluate model on validation/test data
if evaluate:
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_data, y_test)
    st.write(f'Test Accuracy: {test_acc:.2f}')

    y_pred = model.predict(test_data)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report and confusion matrix
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred_class, target_names=class_names))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(plt.gcf())

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    st.write(f'ROC-AUC Score: {roc_auc:.2f}')

# Plot training history
if plot_history:
    st.subheader("Training History")
    try:
        hist = pd.read_csv(os.path.join(base_dir, 'history.csv'))  # Update path as necessary
        st.write("### Training and Validation Loss")
        st.line_chart(hist[['loss', 'val_loss']])
        st.write("### Training and Validation Accuracy")
        st.line_chart(hist[['accuracy', 'val_accuracy']])
    except FileNotFoundError:
        st.write('Training history file not found.')

# Save the model
if save_model:
    model.save('CNN_image_classification_model.h5')
    st.write("Model saved successfully!")

# Complete progress bar
progress_bar.progress(100)  # Simulate completion
time.sleep(1)  # Optional: Pause for a moment before hiding progress bar
progress_bar.empty()

# Footer
st.write("This app uses a Convolutional Neural Network (CNN) to classify brain tumors.")
