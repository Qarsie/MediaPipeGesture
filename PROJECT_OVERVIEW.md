# MediaPipe ASL Gesture Recognition Project

## 🎯 Project Overview

This project implements a comprehensive real-time American Sign Language (ASL) recognition system using MediaPipe for hand landmark detection, deep learning for feature extraction and classification, and FastAPI for deployment. The system can recognize all 26 ASL alphabet letters (A-Z) from live camera feed or uploaded images.

## 🏗️ Architecture

The project follows a multi-stage machine learning pipeline:

1. **Data Collection & Preprocessing** → MediaPipe hand landmark extraction
2. **Feature Engineering** → Autoencoder for dimensionality reduction
3. **Classification** → 1D CNN with PSO-optimized hyperparameters
4. **Deployment** → FastAPI web service with WebSocket support

## 📁 Project Structure

```
MediaPipeGesture/
├── 📄 main.py                     # FastAPI web service entry point
├── 📄 realtime_predict.py         # Real-time ASL recognition class
├── 📄 setup.py                    # Package setup configuration
├── 📄 README.md                   # Project documentation
├── 📄 report.log                  # Training/execution logs
│
├── 📂 asl_dataset/                # Raw image dataset
│   ├── A/, B/, C/, ..., Z/        # 26 folders for each ASL letter
│   └── [50 images per letter]     # Training images
│
├── 📂 data_preparation/           # Data preprocessing scripts
│   ├── 📄 extract_landmarks.py    # MediaPipe landmark extraction
│   └── 📄 augment_landmarks.py    # Data augmentation (noise, rotation)
│
├── 📂 landmarks/                  # Processed landmark data
│   ├── 📄 asl_landmarks.csv       # Original extracted landmarks
│   ├── 📄 augmented_landmarks.csv # Augmented dataset
│   └── 📄 A.csv, B.csv, ...       # Individual letter landmarks
│
├── 📂 models/                     # Model definitions and saved weights
│   ├── 📄 autoencoder.py          # Autoencoder architecture
│   ├── 📄 classifier.py           # 1D CNN classifier
│   ├── 📄 pso_optimizer.py        # Particle Swarm Optimization
│   ├── 📄 encoder.h5              # Trained encoder weights
│   ├── 📄 classifier.h5           # Trained classifier weights
│   └── 📄 autoencoder.h5          # Full autoencoder weights
│
├── 📂 training/                   # Training scripts
│   ├── 📄 train_autoencoder.py    # Train feature extractor
│   └── 📄 train_classifier.py     # Train classifier with PSO
│
└── 📂 utils/                      # Utility functions
    ├── 📄 preprocess.py           # Landmark normalization
    └── 📄 visualization.py        # Plotting and drawing functions
```

## 🔬 Technical Details

### Data Pipeline

#### 1. **Landmark Extraction** (`data_preparation/extract_landmarks.py`)
- Uses MediaPipe Hands to detect 21 3D hand landmarks (x, y, z coordinates)
- Processes 1,300 images (50 per letter × 26 letters)
- Outputs 63-dimensional feature vectors (21 landmarks × 3 coordinates)

#### 2. **Data Augmentation** (`data_preparation/augment_landmarks.py`)
- **Noise Augmentation**: Adds Gaussian noise (σ = 0.01) to landmarks
- **Rotation Augmentation**: Applies random rotations (-15° to +15°)
- Increases dataset size by ~8x for improved generalization

#### 3. **Normalization** (`utils/preprocess.py`)
```python
# Centers landmarks at wrist (landmark 0)
centered = landmarks - landmarks[:, 0:1, :]
# Scales by palm size (wrist to middle finger MCP distance)
normalized = centered / palm_size
```

### Model Architecture

#### 1. **Autoencoder** (`models/autoencoder.py`)
```
Input (63D) → Dense(128) → Dense(64) → Bottleneck(32D) → Dense(64) → Dense(128) → Output(63D)
```
- **Purpose**: Dimensionality reduction and feature learning
- **Latent Space**: 32 dimensions (compressed representation)
- **Regularization**: L1/L2 regularization to prevent overfitting

#### 2. **1D CNN Classifier** (`models/classifier.py`)
```
Input(32,1) → Conv1D(filters) → MaxPool1D → Dropout → Conv1D(filters×2) → GlobalMaxPool → Dense(128) → Dense(26)
```
- **Input**: 32D encoded features
- **Output**: 26 classes (A-Z probability distribution)
- **Hyperparameters**: Optimized using PSO

#### 3. **Particle Swarm Optimization** (`models/pso_optimizer.py`)
- **Optimizes**: Filter count, dropout rate, learning rate
- **Search Space**: 
  - Filters: [32, 256]
  - Dropout: [0.1, 0.5]
  - Learning Rate: [10⁻⁴, 10⁻²]
- **Objective**: Maximize validation accuracy

### Real-time Recognition

#### **ASLRecognizer Class** (`realtime_predict.py`)
- **MediaPipe Integration**: Real-time hand detection
- **Preprocessing**: Automatic landmark normalization
- **Inference Pipeline**: Landmark → Encoder → Classifier → Prediction
- **Confidence Threshold**: 0.7 (adjustable)

## 🚀 Deployment

### **FastAPI Web Service** (`main.py`)

#### **Endpoints:**
1. **POST `/predict-image/`**
   - Upload image file for ASL prediction
   - Returns: `{"label": "A", "confidence": 0.95}`

2. **WebSocket `/ws/recognize`**
   - Real-time video stream processing
   - Input: Base64-encoded JPEG frames
   - Output: Continuous predictions with confidence scores

#### **CORS Configuration:**
- Allows cross-origin requests for web integration
- Configurable origins for security

## 🔧 Key Features

### **Performance Optimizations**
- **MediaPipe**: Hardware-accelerated hand detection
- **Lightweight Models**: Encoder (32D) + CNN for fast inference
- **Batch Processing**: Efficient landmark processing
- **Early Stopping**: Prevents overfitting during training

### **Robustness Features**
- **Translation Invariance**: Wrist-centered normalization
- **Scale Invariance**: Palm-size normalization
- **Rotation Tolerance**: Augmentation with rotated samples
- **Noise Resilience**: Gaussian noise augmentation

### **Real-time Capabilities**
- **Low Latency**: ~50ms inference time
- **High Accuracy**: >90% on test set
- **Confidence Scoring**: Reliability assessment
- **Visual Feedback**: Hand landmark overlay

## 📊 Dataset Information

- **Size**: 1,300 base images → ~10,000 augmented samples
- **Classes**: 26 ASL alphabet letters (A-Z)
- **Format**: RGB images → 3D hand landmarks
- **Split**: 80% training, 20% validation
- **Augmentation**: 8x increase through noise and rotation

## 🛠️ Dependencies

### **Core Libraries**
- **TensorFlow/Keras**: Deep learning framework
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Image processing
- **FastAPI**: Web service framework
- **NumPy/Pandas**: Data manipulation

### **Optimization**
- **PySwarms**: Particle Swarm Optimization
- **Scikit-learn**: Data splitting and preprocessing

### **Deployment**
- **Uvicorn**: ASGI server
- **WebSockets**: Real-time communication

## 🎯 Use Cases

### **Educational Applications**
- ASL learning platforms
- Sign language practice tools
- Accessibility training

### **Communication Aids**
- Real-time ASL-to-text translation
- Video call sign language recognition
- Mobile accessibility apps

### **Research Applications**
- Gesture recognition research
- Computer vision benchmarking
- Human-computer interaction studies

## 🔄 Training Workflow

1. **Data Preparation**
   ```bash
   python data_preparation/extract_landmarks.py
   python data_preparation/augment_landmarks.py
   ```

2. **Model Training**
   ```bash
   python training/train_autoencoder.py
   python training/train_classifier.py
   ```

3. **Deployment**
   ```bash
   python main.py  # or uvicorn main:app
   ```

## 📈 Performance Metrics

- **Accuracy**: >90% on validation set
- **Inference Speed**: ~50ms per frame
- **Model Size**: <10MB total (encoder + classifier)
- **Memory Usage**: <100MB during inference

## 🔮 Future Enhancements

### **Technical Improvements**
- **Dynamic Gestures**: Support for motion-based signs
- **Multi-hand Detection**: Two-handed sign recognition
- **Temporal Modeling**: LSTM/Transformer for sequence recognition

### **Deployment Scaling**
- **Mobile Integration**: TensorFlow Lite conversion
- **Edge Computing**: ONNX model optimization
- **Cloud Deployment**: Docker containerization

### **Dataset Expansion**
- **More Sign Languages**: International sign language support
- **Continuous Signs**: Full sentence recognition
- **User Adaptation**: Personalized model fine-tuning

## 🤝 Contributing

This project serves as a comprehensive example of:
- Computer vision pipeline development
- Deep learning model optimization
- Real-time application deployment
- Web service architecture

The modular design allows for easy extension and modification of individual components while maintaining system integrity.

---

**Project Type**: Final Year Project (FYP) - Computer Vision & Machine Learning  
**Domain**: Accessibility Technology & Sign Language Recognition  
**Status**: Production-ready with real-time capabilities
