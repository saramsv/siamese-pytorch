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

    def __init__(self, imgs_info, transform=None):
        super(DecomTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.imgs, self.anns, self.num_imgs, self.classes = self.loadToMem(imgs_info)

    def loadToMem(self, imgs_info):
        print("begin loading training dataset to memory")
        df = pd.read_json(imgs_info, lines=True)
        img_paths = list(df['fpath_img'].values)
        anns = df['fpath_segm'].values
        ann_paths = ["/data/sara/semantic-segmentation-pytorch/" + ann for ann in anns]
	
        print("finish loading training dataset to memory")
        return img_paths, ann_paths, len(img_paths), list(df.columns[6:])

    def iou(self, label1, label2):
        tmp_label2 = cv2.resize(label2, (int(label1.shape[1]), int(label1.shape[0])))
        label1[np.where(label1 == 0)] = 255 # to exclude the bg from intersection
        intersection = np.where(label1[:,:,0] == tmp_label2[:,:,0])[0].shape[0]
        union = np.where(label1[:,:,0] != 255 )[0].shape[0] + \
                        np.where(tmp_label2[:,:,0] != 0 )[0].shape[0] - \
                        intersection
        return intersection/union


    def __len__(self):
        return  21000000
	
    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None

        img1_index = random.randint(0, self.num_imgs - 1)
        img2_index = random.randint(0, self.num_imgs - 1)
        #try:
        image1 = Image.open(self.imgs[img1_index]).convert('L')
        ##image1 = image1.resize((105, 105), Image.BILINEAR)
        image2 = Image.open(self.imgs[img2_index]).convert('L')
        ##image2 = image2.resize((105, 105), Image.BILINEAR)
        #except:
        #    print(f"idx1: {img1_index}, idx2: {img2_index}, len imgs: {len(self.imgs)}")
        #    import bpython
        #    bpython.embed(locals())
        #    exit()
        ann1 = cv2.imread(self.anns[img1_index])
        ann2 = cv2.imread(self.anns[img2_index])
        IoU = self.iou(ann1, ann2)
        if IoU > 0.45:
            label = 1
        else:
            label = 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        #print(f"iou: {IoU} and label is {label}")
        #print(image1.shape, image2.shape, torch.from_numpy(np.array([label], dtype=np.float32)))
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class DecomTest(Dataset):

    def __init__(self, imgs_info, transform=None, times=200, way=20):
        np.random.seed(1)
        super(DecomTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.imgs, self.anns, self.num_imgs = self.loadToMem(imgs_info)

    def loadToMem(self, imgs_info):
        print("begin loading training dataset to memory")
        df = pd.read_json(imgs_info, lines=True)
        img_paths = list(df['fpath_img'].values)
        anns = df['fpath_segm'].values
        ann_paths = ["/data/sara/semantic-segmentation-pytorch/" + ann for ann in anns]
	
        print("finish loading training dataset to memory")
        return img_paths, ann_paths, len(img_paths)

    def iou(self, label1, label2):
        tmp_label2 = cv2.resize(label2, (int(label1.shape[1]), int(label1.shape[0])))
        label1[np.where(label1 == 0)] = 255 # to exclude the bg from intersection
        intersection = np.where(label1[:,:,0] == tmp_label2[:,:,0])[0].shape[0]
        union = np.where(label1[:,:,0] != 255 )[0].shape[0] + \
                        np.where(tmp_label2[:,:,0] != 0 )[0].shape[0] - \
                        intersection
        return intersection/union

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        label = None
        img1_index = random.randint(0, self.num_imgs - 1)
        img2_index = random.randint(0, self.num_imgs - 1)
        image1 = Image.open(self.imgs[img1_index]).convert('L')
        image2 = Image.open(self.imgs[img2_index]).convert('L')
        ann1 = cv2.imread(self.anns[img1_index])
        ann2 = cv2.imread(self.anns[img2_index])
        IoU = self.iou(ann1, ann2)
        if IoU > 0.45:
            label = 1
        else:
            label = 0
        '''
        import bpython
        bpython.embed(locals())
        exit()
        '''

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


# test
if __name__=='__main__':
    odecomTrain = DecomTrain('/usb/seq_data_for_mit_code/', 30000*8)
    print(odecomTrain)
