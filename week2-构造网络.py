import torch
from torch import nn, optim    #导入pytorch库以及它的nn和optim模块

# 定义线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):    #__init__:接受两个参数  /  input_size :输入特征纬度  / num_class:分类数目
        super(LinearClassifier, self).__init__()
        # 初始化时定义模型结构，这里就是一个简单的线性层
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        # 实现前向传播逻辑
        return self.linear(x)

# 假设输入特征的维度为10，分类数为3
input_size = 10
num_classes = 3

# 创建模型实例
model = LinearClassifier(input_size, num_classes)

# 定义损失函数，这里使用交叉熵损失函数，适用于多分类问题
loss_fn = nn.CrossEntropyLoss()   # CrossEntropuloss结合了LogSoftmax和NLLLoss的损失函数，适用于多分类问题。

# 选择优化器，这里使用随机梯度下降（SGD）
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 假设我们有一些输入数据和对应的标签
inputs = torch.randn(5, input_size)  # 生成5个样本，每个样本有10个特征，值为标准正态分布的随机数
labels = torch.randint(0, num_classes, (5,))  # 生成5个样本的标签，介于0到num_class-1之间的随机数

# 前向传播
outputs = model(inputs)
loss = loss_fn(outputs, labels)

# 反向传播和优化
optimizer.zero_grad()  # 清空梯度
loss.backward()  # 计算梯度 损失相对于每个参数的梯度
optimizer.step()  # 更新参数

print(f'Loss: {loss.item()}')  # 输出损失值