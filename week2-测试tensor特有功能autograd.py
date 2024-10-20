import torch

# 创建一个需要梯度计算的Tensor
x = torch.tensor([1.0], requires_grad=True)

# 对Tensor进行一系列操作
y = x ** 2

# 计算y关于x的梯度
y.backward()

# 打印x的梯度
print(x.grad)  #输出：tensor（[2.])

# 创建另一个需要梯度计算的Tensor
z = torch.tensor([2.0], requires_grad=True)

# 对Tensor进行更复杂的操作
w = z ** 3

# 计算w关于z的梯度
w.backward()

# 打印z的梯度
print(z.grad)  #输出：tensor([12.])