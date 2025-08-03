import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = os.path.join("model", "best_calorify.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Sesuaikan dengan arsitektur model
CLASS_NAMES = ["Rendah", "Sedang", "Tinggi"]

def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    preds = model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    label = f"Class {class_idx}"  # Label sementara
    return label, confidence


