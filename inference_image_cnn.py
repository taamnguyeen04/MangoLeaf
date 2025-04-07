import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.models import resnet34, ResNet34_Weights
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train NN model")
    parser.add_argument("--image_path", "-p", type=str, default=r"C:\Users\tam\Documents\Data\MangoLeafBD Dataset\Healthy\20211231_123327 (Custom).jpg")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/leaf")
    args = parser.parse_args()

    return args


def inference(args):
    classes = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (args.image_size, args.image_size))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # equivalent to ToTensor() from Pytorch
    image = image / 255.

    # equivalent to Normalize()
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image).float().to(device)

    model = resnet34()
    model.fc = nn.Linear(in_features=512, out_features=8)
    checkpoint = torch.load(os.path.join(args.checkpoint_path, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        prob = softmax(output)
        predicted_prob, predicted_class = torch.max(prob, dim=1)
        # print(predicted_prob, predicted_class)
        # print("The image is about {}".format(classes[predicted_class]))
        score = predicted_prob[0]*100
        cv2.imshow("{} with confident score of {:0.2f}%".format(classes[predicted_class[0]], score), cv2.imread(args.image_path))
        cv2.waitKey(0)



if __name__ == '__main__':
    args = get_args()
    inference(args)
