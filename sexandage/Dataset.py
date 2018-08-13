import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
from augmentations import RandomRotate
from augmentations import *
from PIL import Image
from PIL import ImageEnhance


class PersonDataset(data.Dataset):
    def __init__(self, root, transform=None, train=True, augmentations=None):
        self.data_list = []
        self.age_list = []
        self.sex_list = []
        self.state = train
        self.age_type = {"personalLess15":0, "personalLess30":1, "personalLess45":2, "personalLess60":3,"personalLarger60":4}
        self.sex_type = {"personalMale":0, "personalFemale":1}
        self.augmentations = augmentations
        # self.Rotate = RandomRotate(10)
        # self.ColorAug = RandomColorAug(1.5)
        # self.SharpAug = RandomSharpAug(3)
        # self.Gaussian = RandomGaussian(2)
        # self.Contrast = RandomContrast(1.5)
        # self.Bright = RandomBright(1.5)
        if train:
            with open(root + "/train.txt") as f:
                for line in f:
                    item = line.strip().split(" ")
                    image_path = item[0]
                    age = item[1]
                    sex = item[2]
                    label_age = self.age_type[age]
                    label_sex = self.sex_type[sex]
                    self.age_list.append(label_age)
                    self.sex_list.append(label_sex)
                    self.data_list.append(image_path)
        else:
            with open(root + "test.txt") as f:
                for line in f:
                    item = line.strip().split(" ")
                    image_path = item[0]
                    self.data_list.append(image_path)

        self.transform = transform

    def __getitem__(self, item):
        img = self.data_list[item]
        age_label = None
        sex_label = None
        image_name = img.split('/')[-1]
        img = cv2.imread(img)
        if self.state:
            # img = self.augmentations(img)
            # img = self.ColorAug(img)
            # img = self.SharpAug(img)
            # img = self.Rotate(img)
            # img = self.Gaussian(img)
            # img = self.Bright(img)
            # img = self.Contrast(img)
            age_label = self.age_list[item]
            sex_label = self.sex_list[item]
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (0, 1, 2))
        if self.transform is not None:
            img = self.transform(img)
        if self.state:
            return img, (age_label, sex_label)
        else:
            return img, image_name

    def __len__(self):
        return len(self.data_list)