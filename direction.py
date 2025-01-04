import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_d11(nn.Module):
    def __init__(self):
        super(Conv_d11, self).__init__()
        kernel = [[-0.5, 0, 0],
                  [0, 1, 0],
                  [0, 0, -0.5]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d12(nn.Module):
    def __init__(self):            
        super(Conv_d12, self).__init__()
        kernel = [[0, -0.5, 0],
                  [0, 1, 0],
                  [0, -0.5, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d13(nn.Module):
    def __init__(self):            
        super(Conv_d13, self).__init__()
        kernel = [[0, 0, -0.5],
                  [0, 1, 0],
                  [-0.5, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d14(nn.Module):
    def __init__(self):            
        super(Conv_d14, self).__init__()
        kernel = [[0, 0, 0],
                  [-0.5, 1, -0.5],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)
