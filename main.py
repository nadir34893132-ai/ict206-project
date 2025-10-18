from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import requests
import os

app = FastAPI()

#frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOW_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "cnn_model.h5")



def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def home():
    return {"message": "FastAPI  is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    result = "Malignant" if prediction[0][0] > 0.5 else "Benign"
    return {"prediction": result, "confidence": float(prediction[0][0])}
