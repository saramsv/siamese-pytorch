import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
import pandas as pd
import cv2


class DecomTrain(Dataset):

    def __init__(self, sim_img_info, dissim_img_info, transform=None):
        super(DecomTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.sim_data, self.dissim_data, self.num_pairs = self.loadToMem(sim_img_info, dissim_img_info)

    def loadToMem(self, sim_img_info, dissim_img_info):
        print("begin loading training dataset to memory")
        agrees = [0, 90, 180, 270]
        num = 0
        sim_data = pd.read_json(sim_img_info, lines=True)
        dissim_data = pd.read_json(dissim_img_info, lines=True)
        assert sim_data.shape == dissim_data.shape
        print("finish loading training dataset to memory")
        return sim_data, dissim_data, sim_data.shape[0]
		
    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        label = None
        image1 = None
        image2 = None
        idx = random.randint(0, self.num_pairs - 1)

        def read_data(df, idx):
            image1 = Image.open(df.iloc[idx]['fpath_img'][0]).convert('L')
            image2 = Image.open(df.iloc[idx]['fpath_img'][1]).convert('L')
            label = df.iloc[idx]['iou']
            if label > 0.6:
                label = 1
            else:
                label = 0
            return image1, image2, label

        # get image from same class
        if index % 2 == 1:
            image1, image2, label = read_data(self.sim_data, idx)
        # get image from different class
        else:
            image1, image2, label = read_data(self.dissim_data, idx)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

class DecomTest(Dataset):

    def __init__(self, test_pair_info, transform=None, times=200, way=20):
        np.random.seed(1)
        super(DecomTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.pair_data, self.num_pairs = self.loadToMem(test_pair_info)

    def loadToMem(self, test_pair_info):
        print("begin loading training dataset to memory")
        agrees = [0, 90, 180, 270]
        num = 0
        pair_data = pd.read_json(test_pair_info, lines=True)
        print("finish loading training dataset to memory")
        return pair_data, pair_data.shape[0]


    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        label = None
        image1 = None
        image2 = None
        idx = random.randint(0, self.num_pairs - 1)

        image1 = Image.open(self.pair_data.iloc[idx]['fpath_img'][0]).convert('L')
        image2 = Image.open(self.pair_data.iloc[idx]['fpath_img'][1]).convert('L')
        label = self.pair_data.iloc[idx]['iou']
        if label > 0.6:
            label = 1
        else:
            label = 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


# test
if __name__=='__main__':
    DecomTrain = DecomTrain('/data/sara/DecompositionFeatureSegmentation/data/bodyparts_csv/train_iou.6.7.8.9.odgt',\
	"/data/sara/DecompositionFeatureSegmentation/data/bodyparts_csv/train_iou.less_than.6_subsamples.odgt",\
	30000*8)
    DecomTest = DecomTest("/data/sara/DecompositionFeatureSegmentation/data/bodyparts_csv/siamese_test.odgt")
    import bpython
    bpython.embed(locals())
    exit()
    print(omniglotTrain)
