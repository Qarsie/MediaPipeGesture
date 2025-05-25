import numpy as np

def normalize_landmarks(landmarks):
    """Normalize landmarks to be centered at wrist and scaled by palm size"""
    landmarks = landmarks.reshape(-1, 21, 3)
    
    # Center around wrist (landmark 0)
    centered = landmarks - landmarks[:, 0:1, :]
    
    # Scale by palm size (distance between wrist and middle finger MCP)
    scale = np.linalg.norm(centered[:, 9:10, :], axis=2)  # MCP joint
    normalized = centered / (scale + 1e-8)[:, :, np.newaxis]
    
    return normalized.reshape(-1, 63)

def split_data(X, y, test_size=0.2):
    """Custom train-test split preserving class distribution"""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    return next(sss.split(X, y))