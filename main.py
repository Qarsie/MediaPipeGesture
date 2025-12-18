"""
ASL Recognition API
FastAPI service for real-time American Sign Language recognition
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64

from realtime_predict import ASLRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global recognizer instance
recognizer: Optional[ASLRecognizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app - loads models on startup"""
    global recognizer
    logger.info("Loading ASL recognition models...")
    try:
        recognizer = ASLRecognizer()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    yield
    # Cleanup on shutdown
    logger.info("Shutting down ASL Recognition API...")


# FastAPI app with metadata
app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language recognition using MediaPipe and Deep Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
# In production, replace "*" with your actual Vercel domain
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ASL Recognition API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict-image/",
            "websocket": "/ws/recognize",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return {
        "status": "healthy",
        "model_loaded": recognizer is not None,
        "service": "asl-recognition-api"
    }

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """
    POST endpoint for single image prediction.
    Accepts a JPEG/PNG image and returns the predicted ASL label and confidence.
    
    - **file**: Image file (JPEG, PNG, or WebP)
    - **Returns**: Predicted ASL letter and confidence score
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 10MB")
    
    try:
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        label, confidence = recognizer.predict_from_image(frame)
        
        return {
            "success": True,
            "label": label,
            "confidence": confidence,
            "message": f"Detected: {label}" if label else "No hand detected"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.websocket("/ws/recognize")
async def recognize_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASL recognition.
    Accepts base64-encoded JPEG frames, returns label + confidence.
    
    Connection: ws://host/ws/recognize
    Send: base64 encoded JPEG image
    Receive: {"label": "A", "confidence": 0.95, "success": true}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")
    
    frames_processed = 0
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection keep-alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        "success": False,
                        "error": "Invalid frame data"
                    })
                    continue

                label, confidence = recognizer.predict_from_image(frame)
                frames_processed += 1
                
                await websocket.send_json({
                    "success": True,
                    "label": label,
                    "confidence": confidence,
                    "frame_count": frames_processed
                })
                
            except base64.binascii.Error:
                await websocket.send_json({
                    "success": False,
                    "error": "Invalid base64 encoding"
                })
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_json({
                    "success": False,
                    "error": "Processing failed"
                })
                
    except Exception as e:
        logger.info(f"WebSocket disconnected: {websocket.client} - {e}")
    finally:
        logger.info(f"Total frames processed: {frames_processed}")
        try:
            await websocket.close()
        except:
            pass
