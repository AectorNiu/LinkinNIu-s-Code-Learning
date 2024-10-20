#基于之前构造的网络进行训练
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader#用于处理数据集和数据加载

# 定义线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# 生成数据
def generate_data(num_samples, input_size, num_classes):
    X = torch.randn(num_samples, input_size)
    # 假设规则是根据输入特征的第一个元素的符号来决定类别
    y = (X[:, 0] > 0).long()
    return X, y

# 创建数据集
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 参数设置
input_size = 10
num_classes = 2
num_samples = 1000
batch_size = 32
epochs = 10

# 生成数据
X, y = generate_data(num_samples, input_size, num_classes)

# 创建数据集和数据加载器
dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = LinearClassifier(input_size, num_classes)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training complete")