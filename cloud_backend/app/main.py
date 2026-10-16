import io
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import os
import sys

# Ensure module can be imported whether uvicorn is launched from repo root, cloud_backend, or inside Docker
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from inference import GestureInferenceEngine
except ImportError:
    from app.inference import GestureInferenceEngine

# Global inference engine instance
engine: Optional[GestureInferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the GestureInferenceEngine once on application startup,
    keeping weights warm in GPU/DirectML/CPU memory.
    """
    global engine
    print("[IronMan API] Initializing Gesture Inference Engine...")
    engine = GestureInferenceEngine()
    print(f"[IronMan API] Engine ready on device: {engine.device}")
    yield
    print("[IronMan API] Shutting down...")


app = FastAPI(
    title="IronMan Gesture Inference Cloud API",
    description="Production-grade, low-latency microservice for real-time 2D hand gesture recognition.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "IronMan Gesture Recognition API",
        "status": "online",
        "endpoints": {
            "/health": "Microservice and hardware health check",
            "/predict": "POST single image frame for real-time 2D static pose classification",
            "/docs": "Interactive OpenAPI documentation"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health probe for Kubernetes readiness/liveness checks and AWS Load Balancer targets.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not yet initialized"
        )
    
    return {
        "status": "healthy",
        "device": str(engine.device),
        "model_2d_ready": engine.model_2d is not None,
        "supported_gestures_2d": engine.LABELS_2D,
        "version": "1.0.0"
    }


@app.post("/predict", tags=["Inference"])
async def predict_2d_gesture(
    file: UploadFile = File(..., description="JPEG or PNG image frame from webcam client"),
    threshold: float = Query(0.85, ge=0.0, le=1.0, description="Confidence threshold for actionable trigger")
):
    """
    Real-time 2D Static Pose Inference Endpoint:
    - Accepts multipart/form-data image frame
    - Classifies static gesture among 5 classes: [heart, mute, no_gesture, take_photo, thumb_index2]
    - Returns predicted gesture, confidence score, full probability distribution, and inference latency.
    """
    if engine is None or engine.model_2d is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="2D Gesture Model is not loaded on this server."
        )

    # Read uploaded image bytes
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )

    start_time = time.perf_counter()
    try:
        result = engine.predict_2d(image, confidence_threshold=threshold)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "gesture": result["gesture"],
            "confidence": result["confidence"],
            "is_actionable": result["is_actionable"],
            "probabilities": result["probabilities"],
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
