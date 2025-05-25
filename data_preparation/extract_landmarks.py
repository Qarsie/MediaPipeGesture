import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    results = mp_hands.process(image)
    mp_hands.close()
    
    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    return None

def process_dataset(dataset_path, output_file):
    data = []
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for class_name in tqdm(class_folders):
        class_path = os.path.join(dataset_path, class_name)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                landmarks = extract_landmarks(img_path)
                if landmarks:
                    data.append([class_name, *landmarks])
    
    columns = ['class'] + [f'lm_{i//3}_{ax}' for i in range(63) for ax in ['x', 'y', 'z']]
    pd.DataFrame(data, columns=columns).to_csv(output_file, index=False)

if __name__ == "__main__":
    process_dataset('asl_dataset', 'data/landmarks.csv')