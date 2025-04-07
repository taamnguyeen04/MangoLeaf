import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision.models import resnet34

app = FastAPI()

# Danh sách các lớp
CLASSES = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil",
    "Die Back", "Gall Midge", "Healthy",
    "Powdery Mildew", "Sooty Mould"
]

# Load model khi khởi động ứng dụng
model = None
device = None


@app.on_event("startup")
def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo model
    model = resnet34()
    model.fc = nn.Linear(512, len(CLASSES))

    # Load weights từ checkpoint
    checkpoint_path = os.path.join("trained_models/leaf", "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc và tiền xử lý ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                content={"error": "Invalid image file"},
                status_code=400
            )

        # Tiền xử lý giống như code gốc
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0

        # Chuẩn hóa
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Chuyển đổi tensor và đảm bảo là float32
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)  # Thêm .float() ở đây

        # Dự đoán
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)

        # Tạo response
        return JSONResponse(
            content={
                "prediction": CLASSES[preds.item()],
                "confidence": round(conf.item() * 100, 2),
                "probabilities": {
                    cls: round(prob.item() * 100, 2)
                    for cls, prob in zip(CLASSES, probs.squeeze())
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
