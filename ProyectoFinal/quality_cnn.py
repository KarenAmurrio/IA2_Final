# quality_cnn.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QualityCNN(nn.Module):
    """
    CNN ligera para evaluar la calidad de un recorte de rostro.
    Salida: probabilidad (0..1) de que el rostro tenga calidad suficiente.
    """
    def __init__(self):
        super().__init__()
        # Entrada: [1, 112, 112] en escala de grises
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 112x112 -> pool -> 56x56 -> pool -> 28x28
        self.fc1 = nn.Linear(32 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 1)  # 1 neurona (probabilidad)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,16,56,56]
        x = self.pool(F.relu(self.conv2(x)))  # [B,32,28,28]
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))        # salida 0..1
        return x


def preprocess_face(face_img):
    """
    face_img: recorte de rostro en BGR (como viene de OpenCV)
    salida: tensor torch [1, 1, 112, 112] normalizado 0..1
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Redimensionar a 112x112 (coherente con InsightFace)
    gray = cv2.resize(gray, (112, 112))

    # Normalizar a [0, 1]
    gray = gray.astype(np.float32) / 255.0

    # Añadir canal y batch -> [1, 1, 112, 112]
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)

    return tensor
