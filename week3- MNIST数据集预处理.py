#导入必要的库
import torch
from torchvision import datasets, transforms#torchvison.datasets 包含多个常用的数据集，如MNIST  / tramsforms提供一些图像变换功能如缩放裁剪，用于预处理
from torch.utils.data import DataLoader#迭代器，能自动
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换成Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化处理
])

# 下载并加载数据集
train_set = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 使用数据加载器
for images, labels in train_loader:
    # 在这里进行模型的前向传播、计算损失、反向传播等操作
    pass

# 示例：打印出一批图像和标签的大小
for images, labels in train_loader:
    print(images.shape)  # 输出: torch.Size([64, 1, 28, 28])------- 一批有64张图像，图像大小为28*28
    print(labels.shape)  # 输出: torch.Size([64])----标签大小为64/一批中有64个标签
    break  # 打印一批后退出循环
