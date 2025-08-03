import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = os.path.join("model", "best_calorify.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun',
    'capcay', 'gado-gado', 'ikan goreng', 'kerupuk', 'martabak telur', 'mie',
    'nasi goreng', 'nasi putih', 'nugget', 'opor ayam', 'pempek', 'rendang',
    'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu', 'telur', 'tempe',
    'terong balado', 'tumis kangkung', 'udang'
]

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
    label = CLASS_NAMES[class_idx]
    return label, confidence


