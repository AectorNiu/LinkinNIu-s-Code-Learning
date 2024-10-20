import torch
import numpy
tensor = torch.rand(3,3)
arr = numpy.random.random((3,3))
arrFormTensor = tensor.numpy()
print(tensor)
print()
print(arrFormTensor)