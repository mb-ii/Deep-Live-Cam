#!/usr/bin/env python3

import os
import sys
import time
import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import cv2
import insightface
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model instances
face_swapper = None
face_analyser = None

@app.on_event("startup")
async def startup():
    global face_swapper, face_analyser

    logger.info("Loading models...")

    # Initialize face analyser
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    # Initialize face swapper
    model_path = "models/inswapper_128_fp16.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    face_swapper = insightface.model_zoo.get_model(model_path)
    logger.info("Models loaded successfully")

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": face_swapper is not None}

@app.post("/swap_face")
async def swap_face_endpoint(request: dict):
    start = time.time()
    try:
        # Decode image
        image_data = base64.b64decode(request['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        temp_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        temp_frame = temp_frame.astype(np.float32)

        # Reconstruct face objects
        source_face = create_face_from_dict(request['source_face'])
        target_face = create_face_from_dict(request['target_face'])

        # Perform face swap
        result = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

        # Encode result
        _, buffer = cv2.imencode('.png', result)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        end = time.time()
        print(f"Time taken: {end - start} seconds")
        return {"result": result_b64}
    except Exception as e:
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_faces")
async def get_faces_endpoint(request: dict):
    try:
        # Decode image
        image_data = base64.b64decode(request['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get faces
        faces = face_analyser.get(frame)

        # Serialize faces
        result_faces = []
        for face in faces:
            result_faces.append({
                'bbox': face.bbox.astype(np.float32).tolist(),
                'kps': face.kps.astype(np.float32).tolist(),
                'det_score': float(face.det_score),
                'embedding': face.embedding.astype(np.float32).tolist(),
                'normed_embedding': face.normed_embedding.astype(np.float32).tolist(),
                'gender': int(face.gender),
                'age': int(face.age)
            })

        return {"faces": result_faces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_face_from_dict(face_dict):
    class Face:
        def __init__(self, data):
            self.bbox = np.array(data['bbox'], dtype=np.float32)
            self.kps = np.array(data['kps'], dtype=np.float32)
            self.det_score = float(data['det_score'])
            self.embedding = np.array(data['embedding'], dtype=np.float32)
            self.normed_embedding = np.array(data['normed_embedding'], dtype=np.float32)
            self.gender = int(data['gender'])
            self.age = int(data['age'])
    return Face(face_dict)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
