from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64

from realtime_predict import ASLRecognizer

app = FastAPI()
recognizer = ASLRecognizer()

# CORS (update origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """
    POST endpoint for single image prediction.
    Accepts a JPEG/PNG image and returns the predicted ASL label and confidence.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    label, confidence = recognizer.predict_from_image(frame)
    return {"label": label, "confidence": confidence}


@app.websocket("/ws/recognize")
async def recognize_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASL recognition.
    Accepts base64-encoded JPEG frames, returns label + confidence.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            label, confidence = recognizer.predict_from_image(frame)
            await websocket.send_json({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"[WebSocket Closed] {e}")
        await websocket.close()
