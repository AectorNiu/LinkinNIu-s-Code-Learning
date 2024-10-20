import torch

# 定义方程
def f(x):
    return x**2 + 3*x + 2

# 创建一个张量并设置 requires_grad=True
x = torch.tensor(2.0, requires_grad=True)

# 计算方程的值
y = f(x)

# 计算导数
y.backward() 

# 获取导数的值
derivative = x.grad
print(f"The derivative of f at x=2 is: {derivative}")