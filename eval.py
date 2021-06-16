import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese, AlexNet_Siamese
import time
import numpy as np
import sys
from collections import deque
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import random

test_path = "/data/sara/DecompositionFeatureSegmentation/data/bodyparts_csv/siamese_test.odgt"

class DecomTest(Dataset):

    def __init__(self, test_pair_info, transform=None):
        np.random.seed(1)
        super(DecomTest, self).__init__()
        self.transform = transform
        self.pair_data, self.num_pairs = self.loadToMem(test_pair_info)

    def loadToMem(self, test_pair_info):
        print("begin loading training dataset to memory")
        agrees = [0, 90, 180, 270]
        num = 0
        pair_data = pd.read_json(test_pair_info, lines=True)
        print("finish loading training dataset to memory")
        return pair_data, pair_data.shape[0]


    def __len__(self):
        return self.num_pairs #self.times * self.way

    def __getitem__(self, index):
        label = None
        image1 = None
        image2 = None
        idx = random.randint(0, self.num_pairs - 1)

        name1 = self.pair_data.iloc[idx]['fpath_img'][0]
        name2 = self.pair_data.iloc[idx]['fpath_img'][1]
        image1 = Image.open(name1).convert('L')
        image2 = Image.open(name2).convert('L')
        label = self.pair_data.iloc[idx]['iou']
        if label > 0.6:
            label = 1
        else:
            label = 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32)), name1, name2


net = AlexNet_Siamese()
net.load_state_dict(torch.load("models_decom/model.pt"))
net.eval()


data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

testSet = DecomTest(test_path, transform=data_transforms)
testLoader = DataLoader(testSet, batch_size=1, shuffle=False)


for _, (test1, test2, l, name1, name2) in enumerate(testLoader, 1):

    test1, test2 = test1, test2
    test1, test2 = Variable(test1), Variable(test2)
    output = net.forward(test1, test2)
    output_sigm = torch.sigmoid(output)
    output_round = torch.round(output_sigm) # convert the prediction to 0 and 1
    preds = output_round.data.cpu().numpy()
    pred = preds[0][0]
    if pred == 1:
         print(f"{name1[0]}, {name2[0]}, {preds[0][0]}")
