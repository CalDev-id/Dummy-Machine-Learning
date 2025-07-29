import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_excel("Datasets/data_rumah.xlsx")

# Ambil fitur
X = df[['LB', 'LT', 'KT', 'KM', 'GRS']].values
y = df[['HARGA']].values

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training model
history = model.fit(X_train, y_train, epochs=500, verbose=0)

# Plot loss selama training
plt.plot(history.history['loss'])
plt.title('Loss Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Prediksi
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

# Bandingkan hasil prediksi
for i in range(len(X_test)):
    print(f"Prediksi Harga: {y_pred[i][0]:.2f} - Harga Asli: {y_actual[i][0]:.2f}")
