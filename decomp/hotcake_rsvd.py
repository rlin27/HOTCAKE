import torch
import numpy as np
import torch.nn as nn
import torchvision
from numpy import linalg as la
import time


class HighTkd2ConvRSvd(nn.Module):
    def __init__(self, conv_nn_module, k11, k12, r31, r32, r4):

        def simple_randomized_torch_svd(M, k=10):
            B = torch.tensor(M).cuda(0)
            m, n = B.size()
            transpose = False
            if m < n:
                transpose = True
                B = B.transpose(0, 1).cuda(0)
                m, n = B.size()
            rand_matrix = torch.rand((n, k), dtype=torch.double).cuda(0)  # short side by k
            Q, _ = torch.qr(B @ rand_matrix)  # long side by k
            Q.cuda(0)
            smaller_matrix = (Q.transpose(0, 1) @ B).cuda(0)  # k by short side
            U_hat, s, V = torch.svd(smaller_matrix, False)
            U_hat.cuda(0)
            U = (Q @ U_hat)

            if transpose:
                return V.transpose(0, 1), s, U.transpose(0, 1)
            else:
                return U, s, V

        def simple_randomized_svd(M, k=10):
            m, n = M.shape
            transpose = False
            if m < n:
                transpose = True
                M = M.T
            rand_matrix = np.random.normal(size=(M.shape[1], k))
            Q, _ = np.linalg.qr(M @ rand_matrix, mode='reduced')
            smaller_matrix = Q.T @ M
            U_hat, s, V = np.linalg.svd(smaller_matrix, full_matrices=False)
            U = Q @ U_hat

            if transpose:
                return V.T, s.T, U.T
            else:
                return U, s, V

        def HighTKD2(conv_nn_module, k11, k12, r31, r32, r4):
            bias = conv_nn_module.bias
            stride = conv_nn_module.stride
            padding = conv_nn_module.padding
            conv = conv_nn_module.weight.detach().numpy()  # [K2, K1, kernel size, kernel size]
            conv = conv.transpose([2, 3, 1, 0])  # [kernel size, kernel size, K1, K2]
            dim_tensor = conv.shape  # [kernel size, kernel size, K1, K2]
            conv = conv.reshape(
                [dim_tensor[0], dim_tensor[1], k11, k12, dim_tensor[3]])  # [kernel size, kernel size, k11, k12, K2]

            conv_k11 = conv.transpose([2, 0, 1, 3, 4])  # [k11, kernel size, kernel size, k12, K2]
            conv_k11 = conv_k11.reshape([k11, -1])  # [k11, D*D*k12*k2]
            print('The rank of mode-k11: {}'.format(r31))
            start = time.clock()
            u3, s3, vt3 = simple_randomized_svd(conv_k11, k11)
            end = time.clock()
            print('The rank decided by randomized SVD: {}'.format(len(s3)))
            print('The time of randomized SVD: {}'.format(end - start))
            U3 = u3[:, 0:r31]  # [k11, r31]

            conv_k12 = conv.transpose([3, 0, 1, 2, 4])  # [k12, kernel size, kernel size, k11, K2]
            conv_k12 = conv_k12.reshape([k12, -1])
            print('The rank of mode-k12: {}'.format(r32))
            start = time.clock()
            u4, s4, vt4 = simple_randomized_svd(conv_k12, k12)
            end = time.clock()
            print('The rank decided by randomized SVD: {}'.format(len(s4)))
            print('The time of randomized SVD: {}'.format(end - start))
            U4 = u4[:, 0:r32]  # [k12, r32]

            conv_k2 = conv.transpose([4, 0, 1, 2, 3])  # [K2, kernel size, kernel size, k11, k12]
            conv_k2 = conv_k2.reshape([dim_tensor[3], -1])
            print('The rank of mode-k2: {}'.format(r4))
            u5, s5, vt5 = la.svd(conv_k2)
            U5 = u5[:, 0:r4]  # [k2, r4]

            conv_c = conv.transpose([2, 0, 1, 3, 4])  # [k11, kernel size, kernel size, k12, K2]
            conv_c = conv_c.reshape([k11, -1])
            conv_c = np.dot(U3.T, conv_c).reshape(
                [r31, dim_tensor[0], dim_tensor[1], k12, dim_tensor[3]])  # [r31, kernel size, kernel size, k12, K2]
            conv_c = conv_c.transpose([3, 1, 2, 0, 4])  # [k12, kernel size, kernel size, r31, K2]
            conv_c = conv_c.reshape([k12, -1])
            conv_c = np.dot(U4.T, conv_c).reshape(
                [r32, dim_tensor[0], dim_tensor[1], r31, dim_tensor[3]])  # [r32, kernel size, kernel size, r31, K2]
            conv_c = conv_c.transpose([4, 1, 2, 3, 0])  # [K2, kernel size, kernel size, r31, r32]
            conv_c = conv_c.reshape([dim_tensor[3], -1])
            conv_c = np.dot(U5.T, conv_c).reshape([r4, dim_tensor[0], dim_tensor[1], r31, r32])
            conv_c = conv_c.transpose([1, 2, 3, 4, 0])  # [kernel size, kernel size, r31, r32, r4]

            conv_k11 = U3.reshape([1, 1, U3.shape[0], r31])  # [1, 1, k11, r31]
            conv_k12 = U4.reshape([1, 1, U4.shape[0], r32])  # [1, 1, k12, r32]
            conv_c = conv_c.reshape(
                [dim_tensor[0], dim_tensor[1], r31 * r32, r4])  # [kernel size, kernel size, r31 * r32, r4]
            conv_k2 = U5.T.reshape([1, 1, r4, U5.shape[0]])  # [1, 1, r4, K2]
            return conv_k11, conv_k12, conv_k2, conv_c, bias, stride, padding

        super().__init__()
        conv_k11, conv_k12, conv_k2, conv_c, bias, stride, padding = HighTKD2(conv_nn_module, k11, k12, r31, r32, r4)
        size1 = conv_k11.shape  # [1, 1, k11, r31]
        size2 = conv_k12.shape  # [1, 1, k12, r32]
        size4 = conv_c.shape  # [kernel size, kernel size, r31 * r32, r4]
        size3 = conv_k2.shape  # [1, 1, r4, K2]
        conv_k11_weight = torch.from_numpy(conv_k11).permute(3, 2, 0,
                                                             1).float()  # [#output channel: r31, #input channel: k11, 1, 1]
        conv_k12_weight = torch.from_numpy(conv_k12).permute(3, 2, 0,
                                                             1).float()  # [#output channel: r32, #input channel: k12, 1, 1]
        conv_c_weight = torch.from_numpy(conv_c).permute(3, 2, 0,
                                                         1).float()  # [#output channel: r4, #input channel: r31*r32, 1, 1]
        conv_k2_weight = torch.from_numpy(conv_k2).permute(3, 2, 0,
                                                           1).float()  # [#output channel: k2, #input channel: r4, 1, 1]
        self.conv_k11 = nn.Conv2d(size1[2], size1[3], size1[0], bias=False)
        self.conv_k12 = nn.Conv2d(size2[2], size2[3], size2[0], bias=False)
        self.conv_k2 = nn.Conv2d(size3[2], size3[3], size3[0], bias=True)
        self.conv_c = nn.Conv2d(size4[2], size4[3], size4[0], stride=stride, padding=padding, bias=False)
        self.conv_k11.weight = nn.Parameter(data=conv_k11_weight, requires_grad=True)
        self.conv_k12.weight = nn.Parameter(data=conv_k12_weight, requires_grad=True)
        self.conv_k2.weight = nn.Parameter(data=conv_k2_weight, requires_grad=True)
        self.conv_k2.bias = nn.Parameter(data=bias, requires_grad=True)
        self.conv_c.weight = nn.Parameter(data=conv_c_weight, requires_grad=True)

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
    highertkd_conv = HighTkd2ConvRSvd(conv_nn_module, k11, k12, r31, r32, r4)
    out = conv_nn_module(x)
    print(out.shape)
    out2 = highertkd_conv(x)
    print(out2.shape)


if __name__ == '__main__':
    test()
