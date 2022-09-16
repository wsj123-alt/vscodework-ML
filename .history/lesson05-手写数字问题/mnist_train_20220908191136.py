import torch
from torch import nn # nn:神经网络相关工作
from torch.nn import functional as F # f: Functional常用函数
from torch import optim # 优化工具包

import torchvision # minist关于视觉，所欲需要导入视觉工具包
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot # 自定义工具包

batch_size = 512 # 一次处理图片数量（minist图片较小，所以可以一次处理多一点）

# step1. load dataset
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data', # 数据集路径
    train=True, # 指定获取训练集的那10k 
    download=True,  # 若没有图片则自动去网上下载
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # 下载的数据是numpy格式，转换成tensor（torch中的一个数据载体）
        torchvision.transforms.Normalize((0.1307, ), (0.3081, )) # 正则化处理，使数据在0附近均匀分布，提高性能
    ])),
                                           batch_size=batch_size,
                                           shuffle=True) # shuffle表示加载的时候将数据随机打散

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data/',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),
                                          batch_size=batch_size,
                                          shuffle=False) # 没必要打散
# for i in iter(list):iter()返回迭代器对象
#   print(i)

# next(迭代器对象，即iter的返回值)

x, y = next(iter(train_loader))
# torch.Size([512, 1, 28, 28]) torch.Size([512]) tensor(-0.4242) tensor(2.8215)
# 512张图，28*28，对应512个结果（y）；由最大值和最小值发现确实在0附近分布，若不做Normalize过程，则分布于0,1
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')

# 创建非线性神经网络
class Net(nn.Module):

    # 初始化函数，网络
    def __init__(self): 
        super(Net, self).__init__()
        
        # xw+b 新建三层，256，64都是根据经验随机决定的
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10) # 10分类
        
    # 创建计算过程，接收一张图片x
    def forward(self, x):
        # x的shape: [b, 1, 28, 28] b张图片
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x)) # self.fc1(x)表示经过第一层，再把H1送到一个非线性的单元F.relu()中
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x)) # 此时的x为第一层的输出
        # h3 = h2w3+b3
        x = self.fc3(x) # 最后一层加不加激活函数取决于具体任务

        return x # 返回输出

net = Net()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

# 测试，训练
# enumerate()函数：
# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# list(enumerate(seasons)) -> [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28 * 28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
