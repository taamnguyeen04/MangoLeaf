# 🍃 Mango Leaf Disease Classification

This project focuses on identifying **diseases in mango leaves** using deep learning techniques, specifically convolutional neural networks (CNNs). It utilizes a publicly available dataset and supports both training and inference on single images.

---

## 📦 Dataset

- 📂 **Dataset Name:** MangoLeaf  
- 🔗 [Dataset on Mendeley](https://data.mendeley.com/datasets/hxsnvwty3r/1)  
- 📥 [Download ZIP](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hxsnvwty3r-1.zip)  
- 📄 [Reference Paper](https://www.sciencedirect.com/science/article/pii/S2352340923000598)

---

## 🩺 Supported Mango Leaf Diseases

The model is trained to classify **8 types of leaf conditions**, including healthy leaves and 7 common mango leaf diseases:

| No. | Class Name         | Description |
|-----|--------------------|-------------|
| 1   | **Anthracnose**    | A fungal disease causing black spots and rotting. |
| 2   | **Bacterial Canker** | Bacterial infection that causes lesions and defoliation. |
| 3   | **Cutting Weevil** | Insect damage leading to irregular cuts on leaves. |
| 4   | **Die Back**       | Drying of twigs and leaves from tip to base. |
| 5   | **Gall Midge**     | Insect-caused galls leading to curling and swelling. |
| 6   | **Healthy**        | Leaf with no visible disease. |
| 7   | **Powdery Mildew** | Fungal infection with white powdery patches. |
| 8   | **Sooty Mould**    | Black fungal growth usually due to sap-sucking insects. |

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Training

To train the model, run:
```bash
python train_cnn.py
```

## 🧪 Inference

To run inference on a single image:
```bash
python inference_image_cnn.py -p "path\to\image.jpg"
```

## 🌐 API Inference (FastAPI)

### ▶️ Khởi động server
Chạy lệnh sau để khởi động FastAPI server:

```bash
uvicorn main:app --reload
```

Sau khi chạy, truy cập API docs tại: http://127.0.0.1:8000/docs

## 📤 Gửi ảnh để dự đoán
- Chọn endpoint POST /predict/

- Upload một file ảnh lá xoài (.jpg, .png, v.v.)

- Nhận lại:

- - Tên bệnh (prediction)

- - Độ tin cậy (confidence)

- - Xác suất của từng lớp (probabilities)

## 📦 Kết quả mẫu (JSON response)
```json
{
  "prediction": "Powdery Mildew",
  "confidence": 99.49,
  "probabilities": {
    "Anthracnose": 0.04,
    "Bacterial Canker": 0.02,
    "Cutting Weevil": 0.03,
    "Die Back": 0.14,
    "Gall Midge": 0.02,
    "Healthy": 0.02,
    "Powdery Mildew": 99.49,
    "Sooty Mould": 0.24
  }
}
```