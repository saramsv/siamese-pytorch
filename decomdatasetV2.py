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
        self.datas, self.num_classes = self.loadToMem(imgs_info)

    def loadToMem(self, imgs_info):
        print("begin loading training dataset to memory")
        datas = {}
        """even though we might have a few classes, categorizing images based on the classes in them might result in many morecategories.
         I consider these categories as the keys for data with the idea that if we pick two images from the same categories we have a better 
		 chance of getting two images with high iou"""
        agrees = [0, 90, 180, 270]
        num = 0
        df = pd.read_csv(imgs_info)
        CLASSES = list(df.columns)[6:]
		# for each row find all rows of images with exact same classes and count howmany of them exist
        #import bpython
        #bpython.embed(locals())
        #exit()
        temp = df.groupby(by=CLASSES).size().reset_index(name='count')
        # get those with at least 2 images
        temp = temp[temp['count'] > 1]
        # do the following for each batch of images that all include the same classes
        for idx, line in temp.iterrows():
            temp2 = line.to_frame().transpose()
            temp2 = temp2.set_index(CLASSES)
            df2 = df[df.apply(lambda row: tuple(row[CLASSES].values) in temp2.index, axis=1)].reset_index()
            datas[num] = [list(df2['fpath_img'].values) ,list(df2['fpath_segm'].values)]
            num += 1
        print("finish loading training dataset to memory")
        return datas, num
		
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
        img1 = None
        img2 = None
        c1 = c2 = 0
        # get image from same class
        if index % 2 == 1:
            c1 = random.randint(0, self.num_classes - 1)
            c2 = c1
        # get image from different class
        else:
            c1 = random.randint(0, self.num_classes - 1)
            c2 = random.randint(0, self.num_classes - 1)
            while c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)

        try:
            idx1 = random.randint(0, len(self.datas[c1][0]) - 1)
            idx2 = random.randint(0, len(self.datas[c2][0]) - 1)
        except:
            import bpython
            bpython.embed(locals())
            exit()

        image1 = Image.open(self.datas[c1][0][idx1]).convert('L')
        image2 = Image.open(self.datas[c2][0][idx2]).convert('L')

        ann1 = cv2.imread("/data/sara/semantic-segmentation-pytorch/" + self.datas[c1][1][idx1])
        ann2 = cv2.imread("/data/sara/semantic-segmentation-pytorch/" + self.datas[c2][1][idx2])

        IoU = self.iou(ann1, ann2)
        if IoU > 0.5:
            label = 1
        else:
            label = 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
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
        df = pd.read_csv(imgs_info)
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
    DecomTrain = DecomTrain('/data/sara/DecompositionFeatureSegmentation/data/bodyparts_csv/train.csv', 30000*8)
    import bpython
    bpython.embed(locals())
    exit()
    print(omniglotTrain)
