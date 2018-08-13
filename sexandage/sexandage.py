import torch
import torchvision
import Resnet as R
import visdom
import torch.nn as nn
import torch.optim as optim
import torch
from  Dataset import PersonDataset
import torch.nn
import os
import numpy as np
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"




def train(epoch, valid_interval=2):
    best_model = ""
    max_score = 0
    for i in range(epoch):
        model_ft.train()
        # adjust_learning_rate(optimizer, i)
        if optimizer.param_groups[0]['lr'] > 1e-3:
            scheduler.step()
        correct1 = 0
        correct2 = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            batch_size = data.size(0)
            age = target[0]
            sex = target[1]
            data, age, sex = data.cuda(), age.cuda(), sex.cuda()
            optimizer.zero_grad()
            output1, output2 = model_ft(data)
            loss1 = criterion1(output1, age)
            loss2 = criterion1(output2, sex)
            loss = loss1 + loss2
            loss.backward()

            viz.line(
                X=np.array(np.array([i*len(trainloader) + batch_idx])),
                Y=np.array(np.array([loss.data[0]])),
                win=epoch_lot,
                update='append')
            optimizer.step()
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            correct1 += pred1.eq(age.data).cpu().sum()
            correct2 += pred2.eq(sex.data).cpu().sum()
            if batch_idx % 150 == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(
                    i, loss.data[0] ))
        print('Train Epoch: {}\tacc1: {:.3f}'.format(i, float(correct1)/len(trainloader.dataset)))
        print('Train Epoch: {}\tacc2: {:.3f}'.format(i, float(correct2) / len(trainloader.dataset)))

if __name__ == '__main__':
    viz = visdom.Visdom()
    epoch_lot = viz.line(
        X=torch.zeros((1,)),
        Y=torch.zeros((1,)),
        opts=dict(
            xlabel='epoch',
            ylabel='Loss',
            title='epoch_loss',
            legend=['epoch_loss'])
    )
    transform_train = transforms.Compose([
        # transforms.Scale((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.429, 0.410, 0.382), (0.287, 0.285, 0.435)),
    ])


    trainset = PersonDataset(root='./yuanmai_data', train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=4)
    model_ft = R.ResNet50(sex_classes=2, age_classes=5)
    # model_ft = nn.DataParallel(model_ft)
    model_ft.cuda()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    stage1 = 19
    # stage2 = 50
    optimizer = optim.SGD(model_ft.parameters(), lr=1e-1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # optimizer = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=1e-4, nesterov=True, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20], gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 25], gamma=0.1)
    train(stage1)
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # train(stage2)
    torch.save(model_ft.state_dict(), "best_xception.pkl")
    # torch.save(model_ft.state_dict(), "ft_xception.pkl")

