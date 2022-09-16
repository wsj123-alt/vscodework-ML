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

# 网络初始化，创建一个网络对象
net = Net()
# optim.SGD->梯度下降的优化器，通过它来更新
# net.parameters()返回[w1, b1, w2, b2, w3, b3]对象，即我们要优化的权值；lr学习率，momentum帮助更好的优化
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

# 测试，训练
# enumerate()函数：
# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# list(enumerate(seasons)) -> [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

# 对整个数据集迭代3次
for epoch in range(3):

    # 对整个数据集迭代一次
    for batch_idx, (x, y) in enumerate(train_loader):
        # 对每个batch迭代一次
        
        # x: [b, 1, 28, 28], y: [512]
        # net网络是一个全连接层/wx+b层，只能接收[b,feature]2维的，所以需要把x打平：[b, 1, 28, 28] => [b, 784]（把整个图片看做一个特征向量，b和784）
        x = x.view(x.size(0), 28 * 28) # x.size(0)表示batch
        # 经过三层网络后变成了 => [b, 10]
        out = net(x)
        # y_onehot -> [b, 10]
        y_onehot = one_hot(y) # 把输出转换成one-hot编码
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        
        # 1.清零梯度
        optimizer.zero_grad()
        # 2.计算梯度
        loss.backward()
        # 3。更新梯度 w' = w - lr*grad
        optimizer.step()

        # 记录loss值   loss是tensor数据类型，loss.item()取其具体值
        train_loss.append(loss.item())

        # 每隔10个batch，打印一下loss值，发现其一直在下降，比较理想
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# 绘制loss变化
plot_curve(train_loss)
# 跳出循环后，we get optimal [w1, b1, w2, b2, w3, b3]

# 准确度测试
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1) # dim=1表示在维度1上面取最大值的索引
    correct = pred.eq(y).sum().float().item() # pred.eq(y)表示直接比较真实值和预测值是否一致，一致返回1，直接加起来，则correct就表示当前batch中正确的数量
    total_correct += correct # 所有batch累加

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
