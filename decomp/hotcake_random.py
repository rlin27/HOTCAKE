import torch
import torch.nn as nn
import torchvision

class HighTkd2ConvRandom(nn.Module):
    def __init__(self, conv_nn_module, k11, k12, r31, r32, r4):
        super().__init__()
        stride = conv_nn_module.stride
        padding = conv_nn_module.stride
        kernel_size = conv_nn_module.weight.shape[2]
        k2 = conv_nn_module.weight.shape[0]
        self.conv_k11 = nn.Conv2d(k11, r31, 1, bias=False)
        self.conv_k12 = nn.Conv2d(k12, r32, 1, bias=False)
        self.conv_k2 = nn.Conv2d(r4, k2, 1, bias=True)
        self.conv_c = nn.Conv2d(r31 * r32, r4, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        k11 = self.conv_k11.weight.data.shape[1]  # [r31, k11, 1, 1]
        k12 = self.conv_k12.weight.data.shape[1]  # [r32, k12, 1, 1]
        x_shape = x.shape  # [batch_size, #input channel, height, width]
        x = x.reshape([x_shape[0], k11, k12, x_shape[2], x_shape[3]])  # [batch_size, k11, k12, height, width]
        x = x.permute([0, 1, 3, 4, 2])  # [batch_size, k11, height, width, k12]
        x = x.reshape([x_shape[0], k11, x_shape[2] * x_shape[3], k12])  # [batch_size, k11, height*width, k12]
        x = self.conv_k11(x)  # [batch_size, r31, height*width, k12]
        x = x.permute([0, 3, 2, 1])  # [batch_size, k12, height*width, r31]
        x = self.conv_k12(x)  # [batch_size, r32, height*width, r31]
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([x_shape[0], -1, x_shape[2], x_shape[3]])  # [batch_size, r31*r32, height, width]
        x = self.conv_c(x)  # [batch_size, r4, height', width']
        out = self.conv_k2(x)  # [batch_size, k2, height', width']
        return out


def test():
    k11 = 16
    k12 = 16
    r31 = 8
    r32 = 8
    r4 = 132
    x = torch.rand([10, 256, 5, 5])
    model = torchvision.models.vgg16(pretrained=True)
    print(model)
    conv_nn_module = model.features[12]
    highertkd_conv = HighTkd2ConvRandom(conv_nn_module, k11, k12, r31, r32, r4)
    out = conv_nn_module(x)
    print(out.shape)
    out2 = highertkd_conv(x)
    print(out2.shape)


if __name__ == '__main__':
    test()
