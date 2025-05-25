import numpy as np
from pyswarms.single import GlobalBestPSO
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from models.classifier import build_classifier, compile_classifier


class PSOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val, input_dim, n_classes):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_dim = input_dim
        self.n_classes = n_classes

    def evaluate(self, swarm):
        results = []
        for hyperparams in swarm:
            try:
                n_filters = int(hyperparams[0])
                dropout = float(np.clip(hyperparams[1], 0.1, 0.5))
                learning_rate = 10 ** float(hyperparams[2])

                model = build_classifier(
                    input_dim=self.input_dim,
                    n_classes=self.n_classes,
                    filters=n_filters,
                    dropout=dropout
                )
                model = compile_classifier(model, learning_rate=learning_rate)

                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                )

                val_acc = history.history.get('val_accuracy', [0])[-1]
                results.append(-val_acc)  # Minimize negative accuracy

            except Exception as e:
                print(f"[ERROR] Failed to train with hyperparams {hyperparams}: {e}")
                results.append(1.0)  # Penalize failed model

        return np.array(results)

    def optimize(self, n_particles=10, iterations=10):
        bounds = (
            np.array([32, 0.1, -4]),   # filters, dropout, log10(lr)
            np.array([256, 0.5, -2])
        )

        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=3,
            bounds=bounds,
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        )

        best_score, best_pos = optimizer.optimize(self.evaluate, iters=iterations)

        return {
            'filters': int(best_pos[0]),
            'dropout': float(np.clip(best_pos[1], 0.1, 0.5)),
            'learning_rate': 10 ** float(best_pos[2])
        }
