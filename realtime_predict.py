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
        self.CLASS_LABELS = [chr(i) for i in range(65, 91)]  # A-Z
        self.CONFIDENCE_THRESHOLD = 0.7
        self.MODEL_DIR = "models"

        # Load models
        self.encoder, self.classifier = self._load_models()

        # Initialize MediaPipe
        self.hands = self._init_mediapipe()

    def _load_models(self):
        """Load trained models"""
        try:
            encoder_path = os.path.join(self.MODEL_DIR, "encoder.h5")
            classifier_path = os.path.join(self.MODEL_DIR, "classifier.h5")

            encoder = tf.keras.models.load_model(encoder_path, compile=False)
            classifier = tf.keras.models.load_model(classifier_path, compile=False)

            print("[INFO] Models loaded successfully.")
            return encoder, classifier
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            sys.exit(1)

    def _init_mediapipe(self):
        """Initialize MediaPipe Hands solution"""
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """Extract landmarks and predict from a frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Extract landmarks from the first detected hand
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark
            ]).flatten()
            landmarks = normalize_landmarks(landmarks.reshape(1, -1))

            # Predict
            encoded = self.encoder.predict(landmarks, verbose=0)
            preds = self.classifier.predict(encoded, verbose=0)[0]

            return results.multi_hand_landmarks[0], preds
        return None, None

    def predict_from_image(self, frame):
        """Public method for single-frame prediction (for FastAPI)"""
        landmarks, preds = self.process_frame(frame)
        if landmarks is not None and preds is not None:
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx]
            if confidence > self.CONFIDENCE_THRESHOLD:
                return self.CLASS_LABELS[pred_idx], float(confidence)
        return None, 0.0

    def run(self):
        """Run real-time recognition with webcam (debug mode)"""
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("ASL Recognition", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror image
            frame = cv2.flip(frame, 1)

            landmarks, preds = self.process_frame(frame)

            if landmarks and preds is not None:
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx]

                if confidence > self.CONFIDENCE_THRESHOLD:
                    label = f"{self.CLASS_LABELS[pred_idx]} ({confidence:.2f})"
                    cv2.putText(frame, label, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    draw_hand_connections(frame, landmarks)

            # Show frame
            cv2.imshow("ASL Recognition", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run()
