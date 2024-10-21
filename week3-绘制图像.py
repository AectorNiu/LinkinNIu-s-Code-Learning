# 导入必要的库
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换成Tensor
])

# 下载并加载数据集
train_set = datasets.MNIST(root='./data', download=True, train=True, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# 获取一批数据
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 创建一个图形和一组子图
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
axes = axes.flatten()  # 将子图轴对象展平为一维数组

# 遍历这批图像和标签，将它们绘制在子图上
for ax, image, label in zip(axes, images, labels):
    # 将图像数据从Tensor转换为numpy数组，并调整其形状
    img = image.numpy().squeeze()  # 移除多余的维度，并转换为numpy数组
    ax.imshow(img, cmap='gray')  # 绘制图像，使用灰色调色板
    ax.set_title(str(label.item()))  # 设置子图标题为对应的标签
    ax.axis('off')  # 关闭坐标轴

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()  # 显示图形