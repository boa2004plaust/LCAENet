from model.LCAENet.resnet2020 import Bottleneck, ResNetCt
from thop import profile
from model.LCAENet.DySample import DySample
from model.LCAENet.direction import *

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.BatchNorm2d(dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  
        y = y.squeeze(-1).permute(0, 2, 1) 
        y = self.conv(y)  
        y = self.sigmoid(y)  
        y = y.permute(0, 2, 1).unsqueeze(-1)  
        return x * y.expand_as(x)  
class Down(nn.Module):
    def __init__(self,
                 inp_num = 1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.PReLU,
                 **kwargs
                 ):
        super(Down, self).__init__()

        stemWidth = int(8)
        self.stem = nn.Sequential(
            normLayer(1, affine=False),
            nn.Conv2d(1, stemWidth*2, kernel_size=3, stride=1, padding=1, bias=False),
            normLayer(stemWidth*2),
            activate()
        )

        self.d11=Conv_d11()
        self.d12=Conv_d12()
        self.d13=Conv_d13()
        self.d14=Conv_d14()
        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                       radix=2, groups=4, bottleneck_width=bottleneck_width,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False, layer_parms=channels, **kwargs)
        self.conv_1 = DepthwiseSeparableConv2d(16, 16, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.activate_1 = nn.PReLU()
        self.activate_2 = nn.Sigmoid()

    def forward(self, x):
        d11 = self.d11(x)
        d12 = self.d12(x)
        d13 = self.d13(x)
        d14 = self.d14(x)
        md = d11 * d13 + d12 * d14
        md = self.activate_2(md)
        x = self.stem(x)
        ret1 = x
        x = self.conv_1(x.mul(md))
        x = self.norm1(x)
        out = x + ret1
        out = self.activate_1(out)
        out = self.down(out)
        return out

class UPCt(nn.Module):
    def __init__(self, channels=[],
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU
                 ):
        super(UPCt, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0],
                      channels[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[1]),
            activate()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1],
                      channels[2],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[2]),
            activate()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2],
                      channels[3],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[3]),
            activate()
        )
        self.dy1 = DySample(channels[1])
        self.dy2 = DySample(channels[2])
        self.dy3 = DySample(channels[3])
    def forward(self, x):
        x1, x2, x3, x4 = x
        out = self.up1(x4)
        out = x3 + self.dy1(out)
        out = self.up2(out)
        out = x2 + self.dy2(out)
        out = self.up3(out)
        out = x1 + self.dy3(out)
        return out

class Head(nn.Module):
    def __init__(self, inpChannel, oupChannel,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 # Dropout = 0.1
                 ):
        super(Head, self).__init__()
        interChannel = inpChannel // 4
        self.head = nn.Sequential(
            nn.Conv2d(inpChannel, interChannel,
                      kernel_size=3, padding=1,
                      bias=False),
            normLayer(interChannel),
            activate(),
            nn.Conv2d(interChannel, oupChannel,
                      kernel_size=1, padding=0,
                      bias=True)
        )

    def forward(self, x):
        return self.head(x)

class EDN(nn.Module):
    def __init__(self):
        super(EDN, self).__init__()

        self.X1 = ECAAttention()
        self.X2 = ECAAttention()
        self.X3 = ECAAttention()
        self.X4 = ECAAttention()

    def forward(self, x):
        x1 ,x2, x3, x4 = x
        x1 = self.X1(x1)
        x2 = self.X2(x2)
        x3 = self.X3(x3)
        x4 = self.X4(x4)
        return [x1, x2, x3, x4]

class LCAENet(nn.Module):
    def __init__(self, ):
        super(LCAENet, self).__init__()

        self.encoding = Down(channels=[16, 32, 64, 128])
        self.decoding = UPCt(channels=[512, 256,128,64])

        self.headSeg = Head(inpChannel=64, oupChannel=1)
        self.DN = EDN()


    def enhance(self, x):
        ret1, ret2, ret3, ret4 = x
        x = self.DN(x)
        x[0] = x[0] + ret1
        x[1] = x[1] + ret2
        x[2] = x[2] + ret3
        x[3] = x[3] + ret4
        return x



    def forward(self, x):
        x = self.encoding(x)
        x = self.enhance(x)
        x = self.decoding(x)
        x = torch.sigmoid(self.headSeg(x))
        return x

if __name__ == '__main__':

    model = LCAENet().cuda()
    inputs = torch.rand(2, 1, 256, 256).cuda()
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')