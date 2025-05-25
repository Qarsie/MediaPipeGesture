import numpy as np
import pandas as pd

def augment_data(df, noise_std=0.01, rotations=5):
    augmented = []
    for _, row in df.iterrows():
        landmarks = row.values[1:].reshape(-1, 3)
        class_label = row['class']
        
        # Original
        augmented.append([class_label, *landmarks.flatten()])
        
        # Noise augmentation
        for _ in range(2):
            noisy = landmarks + np.random.normal(0, noise_std, landmarks.shape)
            augmented.append([class_label, *noisy.flatten()])
        
        # Rotation augmentation
        for angle in np.linspace(-15, 15, rotations):
            rad = np.radians(angle)
            rot_matrix = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad), np.cos(rad), 0],
                [0, 0, 1]
            ])
            rotated = np.dot(landmarks, rot_matrix)
            augmented.append([class_label, *rotated.flatten()])
    
    return pd.DataFrame(augmented, columns=df.columns)

if __name__ == "__main__":
    df = pd.read_csv('landmarks/asl_landmarks.csv')
    augmented_df = augment_data(df)
    augmented_df.to_csv('landmarks/augmented_landmarks.csv', index=False)