from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import numpy as np
import base64
import cv2

from model import DenseNet_CBAM_Capsule
from utils import preprocess
from gradcam import generate_gradcam


# ========================
# FastAPI Setup
# ========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: limiter au domaine
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Device
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# Load Model
# ========================
model = DenseNet_CBAM_Capsule(num_classes=4)

state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

classes = ["Cyst", "Normal", "Stone", "Tumor"]

print("Model loaded successfully")
print("Device:", device)
print("Model eval mode:", not model.training)


# ========================
# CLAHE (IDENTIQUE AU TRAINING)
# ========================
def apply_clahe(pil_image):

    img = np.array(pil_image)

    # RGB → LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(final)


# ========================
# Health Check
# ========================
@app.get("/")
def health():
    return {"status": "Backend running"}


# ========================
# Prediction Endpoint
# ========================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 🔥 IMPORTANT: Apply CLAHE (comme training)
    image = apply_clahe(image)

    # Preprocess (Resize + Normalize)
    input_tensor = preprocess(image).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    confidence = float(probs[0][pred])

    print("Probabilities:", probs.tolist())
    print("Predicted index:", pred)
    print("Confidence:", confidence)

    # ========================
    # Grad-CAM
    # ========================
    cam = generate_gradcam(
        model,
        input_tensor,
        model.features[-1]
    )

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0

    original = np.array(image.resize((224, 224))) / 255.0
    overlay = np.clip(heatmap * 0.4 + original, 0, 1)
    overlay = np.uint8(255 * overlay)

    _, buffer = cv2.imencode(".png", overlay)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "prediction": classes[pred],
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in probs[0]],
        "gradcam": gradcam_base64
    }