"""
Train CNN Classifier WITHOUT PSO Optimization
Uses default hyperparameters to compare with PSO-optimized version
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.classifier import build_classifier, compile_classifier
from utils.preprocess import normalize_landmarks

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    y = pd.factorize(df.iloc[:, 0])[0]  # Convert labels to integers
    X = normalize_landmarks(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("=" * 70)
    print("CNN CLASSIFIER TRAINING - WITHOUT PSO OPTIMIZATION")
    print("Using DEFAULT hyperparameters")
    print("=" * 70)
    
    # Load and encode data
    print("\n[1/4] Loading data...")
    X_train, X_val, y_train, y_val = load_data('landmarks/augmented_landmarks.csv')
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")

    # Load trained encoder model
    print("\n[2/4] Loading pre-trained encoder...")
    encoder = tf.keras.models.load_model('models/encoder.h5', compile=False)
    X_train_enc = encoder.predict(X_train, verbose=0)
    X_val_enc = encoder.predict(X_val, verbose=0)
    print(f"✓ Encoded features shape: {X_train_enc.shape}")

    # Reshape for CNN input
    X_train_enc = X_train_enc[..., np.newaxis]  # shape: (batch, input_dim, 1)
    X_val_enc = X_val_enc[..., np.newaxis]

    # Build classifier with DEFAULT hyperparameters (NO PSO)
    print("\n[3/4] Building classifier with DEFAULT parameters...")
    print("┌" + "─" * 50 + "┐")
    print("│ DEFAULT HYPERPARAMETERS (Manual/No Optimization) │")
    print("├" + "─" * 50 + "┤")
    print("│ Filters:        64 (default)                    │")
    print("│ Dropout:        0.3 (30%)                       │")
    print("│ Learning Rate:  0.001                           │")
    print("└" + "─" * 50 + "┘")
    
    default_params = {
        'filters': 64,
        'dropout': 0.3,
        'learning_rate': 0.001
    }
    
    classifier = build_classifier(
        input_dim=X_train_enc.shape[1],
        n_classes=26,
        filters=default_params['filters'],
        dropout=default_params['dropout']
    )
    classifier = compile_classifier(classifier, learning_rate=default_params['learning_rate'])
    
    print(f"\n✓ Model built successfully")
    print(f"  Total parameters: {classifier.count_params():,}")

    # Train classifier
    print("\n[4/4] Training classifier...")
    print("-" * 70)
    
    history = classifier.fit(
        X_train_enc, y_train,
        validation_data=(X_val_enc, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'models/classifier_no_pso.h5', 
                save_best_only=True, 
                monitor='val_accuracy',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 70)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n📊 FINAL METRICS (Default Hyperparameters - NO PSO):")
    print("┌" + "─" * 68 + "┐")
    print(f"│ Training Accuracy:    {final_train_acc*100:6.2f}%                                    │")
    print(f"│ Validation Accuracy:  {final_val_acc*100:6.2f}%                                    │")
    print(f"│ Best Val Accuracy:    {best_val_acc*100:6.2f}%                                    │")
    print(f"│ Overfitting Gap:      {(final_train_acc - final_val_acc)*100:6.2f}%                                    │")
    print("└" + "─" * 68 + "┘")
    
    print("\n💾 Model saved as: models/classifier_no_pso.h5")
    
    print("\n" + "=" * 70)
    print("To compare with PSO-optimized model, run:")
    print("  python training/compare_models.py")
    print("=" * 70)
