import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp

def plot_landmarks(landmarks, ax=None):
    """Plot 3D hand landmarks"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    landmarks = landmarks.reshape(-1, 3)
    connections = [
        (0,1,2,3,4),         # Thumb
        (0,5,6,7,8),         # Index
        (0,9,10,11,12),      # Middle
        (0,13,14,15,16),     # Ring
        (0,17,18,19,20)      # Pinky
    ]
    
    for finger in connections:
        x = landmarks[finger, 0]
        y = landmarks[finger, 1]
        z = landmarks[finger, 2]
        ax.plot(x, y, z, marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Train Acc')
        ax2.plot(history.history['val_accuracy'], label='Val Acc')
        ax2.set_title('Accuracy')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

def draw_hand_connections(frame, landmarks):
    """Draw hand landmarks and connections"""
    mp_hands = mp.solutions.hands
    h, w = frame.shape[:2]
    
    # Draw landmarks
    for landmark in landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    
    # Draw connections
    for connection in mp_hands.HAND_CONNECTIONS:
        x1, y1 = int(landmarks.landmark[connection[0]].x * w), int(landmarks.landmark[connection[0]].y * h)
        x2, y2 = int(landmarks.landmark[connection[1]].x * w), int(landmarks.landmark[connection[1]].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
