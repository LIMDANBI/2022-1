import os
import cv2
import torch
from torch import nn

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class path_to_img(Dataset):
    def __init__(self, img_path, labels, transform):  # 데이터셋 전처리
        self.img_path = img_path
        self.labels = labels
        self.transform = transform

    def __len__(self):  # 데이터셋 길이 (총 샘플의 수)
        return len(self.img_path)

    def __getitem__(self, idx):  # 데이터셋에서 특정 샘플을 가져옴
        path = os.path.join("/home/danbibibi/jupyter/Project/data", self.img_path.iloc[idx])
        image = Image.open(path).convert('RGB')
        image = self.transform(image) # [N, C, W, H]
        label = self.labels.iloc[idx]
        return image, label