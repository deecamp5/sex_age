import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from Dataset import PersonDataset
import shutil
import visdom
import numpy as np
import os
import cv2
import Resnet as R


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
age_dict = {"0": "personalLess15", "1": "personalLess30", "2": "personalLess45", "3": "personalLess60",
                 "4" : "personalLarger60"}
sex_dict = {"0": "personalMale", "1": "personalFemale"}
age_count = {"personalLess15": 0, "personalLess30": 0, "personalLess45": 0, "personalLess60": 0,
                 "personalLarger60": 0}
sex_count = {"personalMale": 0, "personalFemale": 0}
if __name__ == '__main__':
    transform_test = transforms.Compose([
        # transforms.Scale(224,224),
        transforms.ToTensor(),
        transforms.Normalize((0.429, 0.410, 0.382), (0.287, 0.285, 0.435))
    ])
    model_ft = R.ResNet50(sex_classes=2, age_classes=5)
    tmp = torch.load('./yuanmai_data/epoch5.pkl')
    tmp2 = {}
    for item in tmp.items() :
        key = item[0]
        values = item[1]
        newkey = key[7:]
        tmp2[newkey] = values

    model_ft.load_state_dict(tmp2)
    model_ft.cuda()
    model_ft.eval()
    correct1 = 0
    correct2 = 0
    predict_list = []
    root_dir = "D:\deecamp\code\crop_picture\\6"
    file_list = os.listdir(root_dir)
    for file in file_list:
        img_path = os.path.join(root_dir,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 256))
        # cv2.imshow("222", img)
        # cv2.waitKey()
        img = transform_test(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        sex_output, age_output = model_ft(img)
        pred1 = sex_output.data.max(1)[1]
        pred2 = age_output.data.max(1)[1]
        age = age_dict[str(int(pred2))]
        sex = sex_dict[str(int(pred1))]
        age_count[age] += 1
        sex_count[sex] += 1
        print("age:", age_dict[str(int(pred2))], "sex", sex_dict[str(int(pred1))])
    print(age_count, sex_count)