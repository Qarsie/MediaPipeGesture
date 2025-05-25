import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.classifier import build_classifier, compile_classifier
from models.pso_optimizer import PSOptimizer
from utils.preprocess import normalize_landmarks

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    y = pd.factorize(df.iloc[:, 0])[0]  # Convert labels to integers
    X = normalize_landmarks(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load and encode data
    X_train, X_val, y_train, y_val = load_data('landmarks/augmented_landmarks.csv')

    # Load trained encoder model
    encoder = tf.keras.models.load_model('models/encoder.h5', compile=False)
    X_train_enc = encoder.predict(X_train)
    X_val_enc = encoder.predict(X_val)

    # Reshape for CNN input
    X_train_enc = X_train_enc[..., np.newaxis]  # shape: (batch, input_dim, 1)
    X_val_enc = X_val_enc[..., np.newaxis]

    # PSO Optimization
    optimizer = PSOptimizer(
        X_train_enc, y_train,
        X_val_enc, y_val,
        input_dim=X_train_enc.shape[1],
        n_classes=26
    )
    best_params = optimizer.optimize(n_particles=10, iterations=10)
    print(f"PSO Best Parameters: {best_params}")

    # Train final model using best hyperparameters
    classifier = build_classifier(
        input_dim=X_train_enc.shape[1],
        n_classes=26,
        filters=best_params['filters'],
        dropout=best_params['dropout']
    )
    classifier = compile_classifier(classifier, learning_rate=best_params['learning_rate'])

    classifier.fit(
        X_train_enc, y_train,
        validation_data=(X_val_enc, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('models/classifier.h5', save_best_only=True, monitor='val_accuracy')
        ]
    )

    print("✅ Classifier training complete. Model saved as 'models/classifier.h5'")
