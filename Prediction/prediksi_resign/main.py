import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import re

# Load data
train_df = pd.read_csv('datasets/data1/train.csv')
test_df = pd.read_csv('datasets/data1/test.csv')

train_df['source'] = 'train'
test_df['source'] = 'test'
df = pd.concat([train_df, test_df])

# Tampilkan semua kolom bertipe object (string)
object_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Kolom kategorikal:", object_cols)

def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r'[^\w\s]', '', text)               # hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()          # hapus spasi berlebih
    return text

# Terapkan fungsi cleaning ke semua kolom kategorikal
for col in object_cols:
    df[col] = df[col].astype(str).apply(clean_text)

# Buang kolom ID dan simpan kolom target
df = df.drop(['Employee ID'], axis=1)

# Encode semua kolom kategorikal
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # pastikan bertipe string
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # simpan encoder jika ingin inverse_transform nanti

# Tangani missing values (isi dengan median)
df = df.fillna(df.median(numeric_only=True))

# Pisahkan kembali train dan test
train_df = df[df['source'] == 1].drop(['source'], axis=1)
test_df = df[df['source'] == 0].drop(['source', 'Attrition'], axis=1)

X = train_df.drop('Attrition', axis=1)
y = train_df['Attrition']

# Normalisasi fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# Split untuk validasi
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Buat model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluasi
y_pred = model.predict(X_val)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_val, y_pred_labels))
print("Accuracy:", accuracy_score(y_val, y_pred_labels))

# Prediksi data test
y_test_pred = model.predict(test_scaled)
y_test_labels = (y_test_pred > 0.5).astype(int)

# Simpan hasil prediksi
submission = pd.read_csv('datasets/data1/test.csv')
submission['Attrition_Prediction'] = y_test_labels
submission.to_csv('submission.csv', index=False)
