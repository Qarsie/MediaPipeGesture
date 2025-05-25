import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.autoencoder import build_autoencoder, compile_autoencoder
from utils.preprocess import normalize_landmarks

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    y = df.iloc[:, 0].values
    X = normalize_landmarks(X)
    return train_test_split(X, X, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load data
    X_train, X_val, _, _ = load_data('landmarks/augmented_landmarks.csv')
    
    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder()
    autoencoder = compile_autoencoder(autoencoder)
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint('models/encoder.h5', save_best_only=True)
        ]
    )
    
    print("Autoencoder training complete. Encoder saved.")