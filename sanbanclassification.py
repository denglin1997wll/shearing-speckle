import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from P21model_load import vgg16_class
# from Classnet import Tudui

dataset_transform = transforms.Compose([
    # 中心裁切
    # transforms.CenterCrop(360),
    # 调整大小
    transforms.Resize([224, 224]),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()]
)
# train_dataset = datasets.ImageFolder(root=r"G:\111sanban\sanban_split1\train", transform=dataset_transform)
# test_dataset = datasets.ImageFolder(root=r"D:\111sanban\sanban_split3\test", transform=dataset_transform)

train_dataset = datasets.ImageFolder(root=r"G:\shiyandata\shiyan60split\train", transform=dataset_transform)
test_dataset = datasets.ImageFolder(root=r"G:\shiyandata\shiyan60split\test", transform=dataset_transform)

#获取数据集length
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))
            # img, label = train_dataset[0]
            # print(label)         标签
            # print(img.size())     torch.Size([1, 256, 256])
            # print(img)        tensor类型
#利用DataLoader加载数据集
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

# #创建神经网络

# # # 加载以前的模型
classnet = torch.load(r"G:\shiyandata\shiyan60split\60model\vgg_class19.pth")
#模型在P21model_load里面。可以更改。
# classnet = torch.load("vgg16_class_method1.pth")
classnet = classnet.cuda()


# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.NLLLoss()
loss_fn = loss_fn.cuda()


# 定义优化器
learning_rate = 0.01
optim = torch.optim.Adam(classnet.parameters(), lr=learning_rate)
# optim = torch.optim.SGD(classnet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数

# 记录训练次数
total_train_step = 0
# 记录测试次数='
total_test_step = 0

# 训练轮数
epoch = 50
#记录测试集的loss和准确率：
test_loss = []
test_accuracy = []
train_loss = []

# 添加tensorboard
# writier = SummaryWriter("./logs_vgglassnet_lr1")

for i in range(epoch):
    print("------第{}轮训练开始------".format(i + 1))
    run_loss = 0
    # 训练步骤开始
    classnet.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = classnet(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        run_loss += loss.item()
        total_train_step += 1
        if total_train_step % 100 == 0:  # %   是取余操作，代表每一百次才会输出一次。免得输出太多
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            # writier.add_scalar("train_loss_vgg", loss.item(), total_train_step)

    # with里面的代码没有梯度，保证不被调优

    # 测试步骤开始
    print("测试步骤开始")
    classnet.eval()
    total_test_loss = 0
    total_accuracy = 0
    a0_1 = 0
    a1_0 = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()

            # t1 = time.perf_counter()
            outputs = classnet(imgs)
            # t2 = time.perf_counter()
            # print(t1)
            # print(t2)
            # print(t2 - t1)

            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy


    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))

    train_loss.append(run_loss / train_data_size)
    test_loss.append(total_test_loss)
    test_accuracy.append(total_accuracy / test_data_size)

    # writier.add_scalar("test_accuracy_vgg", total_accuracy / test_data_size, total_test_step)
    # writier.add_scalar("test_loss_vgg", total_test_loss, total_test_step)
    total_test_step += 1
    torch.save(classnet, r"G:\shiyandata\shiyan60split\60model\vgg_class{}.pth".format(i + 1))
    print("模型已保存")

# writier.close()

plt.figure()
plt.plot(test_loss, color="green", label='test_loss')
plt.legend()
plt.show()

print('test_accuracy:', test_accuracy)
print('type(test_accuracy):', type(test_accuracy))
print('test_loss:', test_loss)

