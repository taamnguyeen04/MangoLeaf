from keras.models import load_model

import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# ---- Đường dẫn ảnh cần dự đoán ----
image_path = r"C:\Users\tam\Pictures\unnamed.jpg"

# ---- Tiến hành dự đoán ----
image = cv2.imread(image_path)
if image is None:
    print(f"Không thể đọc ảnh từ: {image_path}")
    exit()

# Hiển thị ảnh gốc
cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize ảnh về (224, 224)
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# Chuyển thành mảng numpy và chuẩn hóa
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
image = (image / 127.5) - 1  # Normalize giống lúc train

# Dự đoán
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

print(f"Class: {class_name} | Confidence: {confidence_score * 100:.2f}%")
