import torch

class SquareFunction(torch.autograd.Function):#我们定义了一个乘方函数
    @staticmethod
    def forward(ctx, input):
        """
        在前向传播中，我们接收到一个输入张量，返回一个输出张量。
        ctx是一个上下文对象，可以用来存储反向传播所需的信息。
        """
        ctx.save_for_backward(input)  # 保存输入张量，以便在反向传播中使用
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播中,我们将ctx作为输入张量用于根据链式法则计算梯度。
        """
        input, = ctx.saved_tensors  # 获取前向传播中保存的输入张量
        grad_input = 2 * input * grad_output  # 根据链式法则计算梯度
        return grad_input

# 测试自定义函数
x = torch.tensor([2.0], requires_grad=True)
y = SquareFunction.apply(x)  # 使用.apply来调用自定义函数
y.backward()  # 计算梯度

print("x:", x)
print("y:", y)
print("x.grad:", x.grad)