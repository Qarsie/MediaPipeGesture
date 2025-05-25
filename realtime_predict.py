"""
Real-time ASL Recognition using MediaPipe and trained models
"""
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from utils.preprocess import normalize_landmarks
from utils.visualization import draw_hand_connections

class ASLRecognizer:
    def __init__(self):
        # Constants
        self.CLASS_LABELS = [chr(i) for i in range(65, 91)]  # A-Z labels
        self.CONFIDENCE_THRESHOLD = 0.7
        self.MODEL_DIR = "models"
        
        # Initialize models
        self.encoder, self.classifier = self._load_models()
        self.hands = self._init_mediapipe()

    def _load_models(self):
        """Load trained models with error handling"""
        try:
            encoder_path = os.path.join(self.MODEL_DIR, "encoder.h5")
            classifier_path = os.path.join(self.MODEL_DIR, "classifier.h5")
            
            encoder = tf.keras.models.load_model(encoder_path, compile=False)
            classifier = tf.keras.models.load_model(classifier_path, compile=False)

            print("Models loaded successfully")
            return encoder, classifier
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def _init_mediapipe(self):
        """Configure MediaPipe hands instance"""
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """Process a single frame for ASL recognition"""
        # Convert and process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Extract and normalize landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] 
                                for lm in results.multi_hand_landmarks[0].landmark]).flatten()
            landmarks = normalize_landmarks(landmarks.reshape(1, -1))
            
            # Predict
            encoded = self.encoder.predict(landmarks, verbose=0)
            preds = self.classifier.predict(encoded, verbose=0)[0]
            
            return results.multi_hand_landmarks[0], preds
        return None, None

    def run(self):
        """Main recognition loop"""
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("ASL Recognition", cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mirror display
            frame = cv2.flip(frame, 1)
            
            # Process frame
            landmarks, preds = self.process_frame(frame)
            
            if landmarks and preds is not None:
                # Get prediction
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx]
                
                if confidence > self.CONFIDENCE_THRESHOLD:
                    # Display prediction
                    label = f"{self.CLASS_LABELS[pred_idx]} ({confidence:.2f})"
                    cv2.putText(frame, label, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Draw landmarks
                    draw_hand_connections(frame, landmarks)
            
            # Display frame
            cv2.imshow("ASL Recognition", frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run()