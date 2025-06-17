1.Landmark Extraction:

Hand landmarks are extracted from images or video frames using the MediaPipe Hands solution.
The extract_landmarks.py script processes a dataset of images to obtain initial 3D hand landmark coordinates.
In real-time, the ASLRecognizer class in realtime_predict.py utilizes MediaPipe to continuously detect and extract these landmarks.

2.Data Preprocessing and Augmentation:

Extracted landmarks undergo normalization to ensure they are centered at the wrist and scaled by the palm size, making them invariant to translation and global scale. This normalization is applied during both training and real-time prediction.
To enhance the model's robustness and generalization, the dataset is augmented by adding random noise and applying rotations to the landmark data.

3.Feature Learning with Autoencoder:

An autoencoder, composed of an encoder and a decoder, is built and trained to learn a compressed, lower-dimensional representation (latent space) of the normalized landmark data.
The encoder component, once trained, is saved and used to transform the high-dimensional landmark data into a more compact feature set for the subsequent classification task.

4.Classification with Convolutional Neural Network (CNN):

A 1D CNN model is constructed to classify the encoded features into specific ASL signs. The CNN includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and dropout for regularization.
The classifier is trained on the encoded landmark data.

5.Hyperparameter Optimization with Particle Swarm Optimization (PSO):

To find the optimal hyperparameters (filters, dropout rate, and learning rate) for the CNN classifier, Particle Swarm Optimization is employed.
The PSOptimizer class iteratively evaluates different hyperparameter combinations by training and validating the classifier, ultimately identifying the best configuration to maximize validation accuracy.

6.Real-time Recognition and Deployment:

The trained encoder and classifier models are loaded by the ASLRecognizer for real-time inference.
A FastAPI application (main.py) provides endpoints for both single image predictions and a WebSocket connection for continuous, real-time video stream recognition, allowing clients to send image frames and receive predicted ASL labels and confidence scores.
