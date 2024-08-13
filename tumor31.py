import streamlit as st
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from PIL import Image
import requests

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Download file at runtime
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        st.success(f"Downloaded file saved to {save_path}")
    else:
        st.error(f"Failed to download file from {url}")

# Load and preprocess images
def load_images_and_labels(data_dir):
    labels_dict = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    images = []
    labels = []

    for label_name in CLASS_NAMES:
        label_dir = os.path.join(data_dir, label_name)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(labels_dict[label_name])

    return np.array(images), np.array(labels)

# Load model
@st.cache_resource
def load_model():
    model_path = 'CNN_image_classification_model_train.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    else:
        st.error("Model file not found!")
        return None

# Image prediction function
def predict_image(model, image):
    img = image.convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction)

    return CLASS_NAMES[predicted_class], confidence_score, prediction

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred_class):
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred):
    fpr = {}
    tpr = {}
    roc_auc = {}

    # One-hot encode the true labels
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)

    plt.figure(figsize=(10, 8))

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred[:, i])
        roc_auc[i] = roc_auc_score(y_true_one_hot[:, i], y_pred[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    return roc_auc

# Streamlit app
def app():
    st.title("Brain Tumor Classification")

    # File download section
    st.sidebar.header("Download Model File")
    url = st.sidebar.text_input("Enter the file URL")
    if st.sidebar.button("Download"):
        if url:
            save_path = "CNN_image_classification_model_train.h5"
            download_file(url, save_path)
        else:
            st.sidebar.warning("Please enter a valid URL")

    # Load model
    model = load_model()

    if model:
        st.write("Model loaded successfully!")

        # Sidebar options
        st.sidebar.header("Options")
        option = st.sidebar.selectbox("Select an option", [
            "Upload Image",
            "Show Classification Report",
            "Show Confusion Matrix",
            "Show ROC Curve",
            "Show Model Summary",
            "Show Training History",
            "Show Test Accuracy"
        ])

        if option == "Upload Image":
            st.sidebar.header("Upload Image")
            uploaded_image_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_image_file:
                st.sidebar.text("Processing...")
                image = Image.open(uploaded_image_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                predicted_class_name, confidence_score, prediction = predict_image(model, image)
                st.sidebar.text(f'Predicted Class: {predicted_class_name}')
                st.sidebar.text(f'Confidence Score: {confidence_score:.4f}')

        elif option == "Show Classification Report":
            test_data_dir = 'Testing'
            test_data, test_labels = load_images_and_labels(test_data_dir)
            y_pred = model.predict(test_data)
            y_pred_class = np.argmax(y_pred, axis=1)
            y_true = test_labels

            st.write("Classification Report:")
            st.text(classification_report(y_true, y_pred_class, target_names=CLASS_NAMES))

        elif option == "Show Confusion Matrix":
            test_data_dir = 'Testing'
            test_data, test_labels = load_images_and_labels(test_data_dir)
            y_pred = model.predict(test_data)
            y_pred_class = np.argmax(y_pred, axis=1)
            y_true = test_labels

            st.write("Confusion Matrix:")
            plot_confusion_matrix(y_true, y_pred_class)

        elif option == "Show ROC Curve":
            test_data_dir = 'Testing'
            test_data, test_labels = load_images_and_labels(test_data_dir)
            y_pred = model.predict(test_data)
            y_true = test_labels

            roc_auc = plot_roc_curve(y_true, y_pred)
            st.write("ROC AUC Scores:")
            for i, score in roc_auc.items():
                st.write(f"Class {CLASS_NAMES[i]}: {score:.2f}")

        elif option == "Show Model Summary":
            with st.spinner("Loading model summary..."):
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.write("Model Summary:")
                st.text("\n".join(model_summary))

        elif option == "Show Training History":
            st.error("Training history is not available. Train the model to view this information.")

        elif option == "Show Test Accuracy":
            test_data_dir = 'Testing'
            test_data, test_labels = load_images_and_labels(test_data_dir)
            y_pred = model.predict(test_data)
            y_pred_class = np.argmax(y_pred, axis=1)
            y_true = test_labels

            test_accuracy = np.mean(y_pred_class == y_true)
            st.write(f"Test Accuracy: {test_accuracy:.2f}")

if __name__ == "__main__":
    app()
