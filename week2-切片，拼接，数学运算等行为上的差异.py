import numpy as np
import torch

# 创建NumPy数组和Tensor
np_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 索引/切片
print("NumPy索引:", np_array[0, 1])  
print("Tensor索引:", tensor[0, 1].item())  

print("NumPy切片:", np_array[:, 1])  
print("Tensor切片:", tensor[:, 1].tolist())  

# 拼接
print("NumPy拼接:", np.concatenate((np_array, np_array), axis=0))  
print("Tensor拼接:", torch.cat((tensor, tensor), dim=0))  

# 数学运算
print("NumPy加法:", np_array + np_array)  
print("Tensor加法:", (tensor + tensor).tolist()) 

print("NumPy乘法:", np_array * np_array)  
print("Tensor乘法:", (tensor * tensor).tolist())  

# 其他操作
print("NumPy转置:", np_array.T)  
print("Tensor转置:", tensor.t())  

print("NumPy求和:", np_array.sum())  
print("Tensor求和:", tensor.sum().item())  