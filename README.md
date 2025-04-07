# 🍃 Mango Leaf Disease Classification

This project focuses on identifying **diseases in mango leaves** using deep learning techniques, specifically convolutional neural networks (CNNs). It utilizes a publicly available dataset and supports both training and inference on single images.

---

## 📦 Dataset

- 📂 **Dataset Name:** MangoLeaf  
- 🔗 [Dataset on Mendeley](https://data.mendeley.com/datasets/hxsnvwty3r/1)  
- 📥 [Download ZIP](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hxsnvwty3r-1.zip)  
- 📄 [Reference Paper](https://www.sciencedirect.com/science/article/pii/S2352340923000598)

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