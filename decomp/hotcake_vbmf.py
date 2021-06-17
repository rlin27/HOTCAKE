import torch
import numpy as np
import torch.nn as nn
import torchvision
import time
from scipy.optimize import minimize_scalar


class HighTkd2ConvVbmf(nn.Module):
    def __init__(self, conv_nn_module, k11, k12):

        def VBMF(Y, cacb, sigma2=None, H=None):
            a = time.clock()
            L, M = Y.shape
            if H is None:
                H = L
            U, s, V = np.linalg.svd(Y)
            U = U[:, :H]
            s = s[:H]
            V = V[:H].T
            residual = 0.
            if H < L:
                residual = np.sum(np.sum(Y ** 2) - np.sum(s ** 2))
            if sigma2 is None:
                upper_bound = (np.sum(s ** 2) + residual) / (L + M)

                if L == H:
                    lower_bound = s[-1] ** 2 / M
                else:
                    lower_bound = residual / ((L - H) * M)

                sigma2_opt = minimize_scalar(VBsigma2, args=(L, M, cacb, s, residual),
                                             bounds=[lower_bound, upper_bound], method='Bounded')
                sigma2 = sigma2_opt.x
                print("Estimated sigma2: ", sigma2)
            thresh_term = (L + M + sigma2 / cacb ** 2) / 2
            threshold = np.sqrt(sigma2 * (thresh_term + np.sqrt(thresh_term ** 2 - L * M)))
            b = time.clock()
            print(b - a)
            pos = np.sum(s > threshold)
            d = np.multiply(s[:pos],
                            1 - np.multiply(sigma2 / (2 * s[:pos] ** 2),
                                            L + M + np.sqrt((M - L) ** 2 + 4 * s[:pos] ** 2 / cacb ** 2)))
            post = {}
            zeta = sigma2 / (2 * L * M) * (
                        L + M + sigma2 / cacb ** 2 - np.sqrt((L + M + sigma2 / cacb ** 2) ** 2 - 4 * L * M))
            post['ma'] = np.zeros(H)
            post['mb'] = np.zeros(H)
            post['sa2'] = cacb * (1 - L * zeta / sigma2) * np.ones(H)
            post['sb2'] = cacb * (1 - M * zeta / sigma2) * np.ones(H)
            delta = cacb / sigma2 * (s[:pos] - d - L * sigma2 / s[:pos])
            post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
            post['mb'][:pos] = np.sqrt(np.divide(d, delta))
            post['sa2'][:pos] = np.divide(sigma2 * delta, s[:pos])
            post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
            post['sigma2'] = sigma2
            post['F'] = 0.5 * (L * M * np.log(2 * np.pi * sigma2) + (residual + np.sum(s ** 2)) / sigma2 - (L + M) * H
                               + np.sum(M * np.log(cacb / post['sa2']) + L * np.log(cacb / post['sb2'])
                                        + (post['ma'] ** 2 + M * post['sa2']) / cacb + (
                                                    post['mb'] ** 2 + L * post['sb2']) / cacb
                                        + (-2 * np.multiply(np.multiply(post['ma'], post['mb']), s)
                                           + np.multiply(post['ma'] ** 2 + M * post['sa2'],
                                                         post['mb'] ** 2 + L * post['sb2'])) / sigma2))
            return U[:, :pos], np.diag(d), V[:, :pos], post

        def VBsigma2(sigma2, L, M, cacb, s, residual):
            H = len(s)
            thresh_term = (L + M + sigma2 / cacb ** 2) / 2
            threshold = np.sqrt(sigma2 * (thresh_term + np.sqrt(thresh_term ** 2 - L * M)))
            pos = np.sum(s > threshold)
            d = np.multiply(s[:pos],
                            1 - np.multiply(sigma2 / (2 * s[:pos] ** 2),
                                            L + M + np.sqrt((M - L) ** 2 + 4 * s[:pos] ** 2 / cacb ** 2)))
            zeta = sigma2 / (2 * L * M) * (
                        L + M + sigma2 / cacb ** 2 - np.sqrt((L + M + sigma2 / cacb ** 2) ** 2 - 4 * L * M))
            post_ma = np.zeros(H)
            post_mb = np.zeros(H)
            post_sa2 = cacb * (1 - L * zeta / sigma2) * np.ones(H)
            post_sb2 = cacb * (1 - M * zeta / sigma2) * np.ones(H)
            delta = cacb / sigma2 * (s[:pos] - d - L * sigma2 / s[:pos])
            post_ma[:pos] = np.sqrt(np.multiply(d, delta))
            post_mb[:pos] = np.sqrt(np.divide(d, delta))
            post_sa2[:pos] = np.divide(sigma2 * delta, s[:pos])
            post_sb2[:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
            F = 0.5 * (L * M * np.log(2 * np.pi * sigma2) + (residual + np.sum(s ** 2)) / sigma2 - (L + M) * H
                       + np.sum(M * np.log(cacb / post_sa2) + L * np.log(cacb / post_sb2)
                                + (post_ma ** 2 + M * post_sa2) / cacb + (post_mb ** 2 + L * post_sb2) / cacb
                                + (-2 * np.multiply(np.multiply(post_ma, post_mb), s)
                                   + np.multiply(post_ma ** 2 + M * post_sa2, post_mb ** 2 + L * post_sb2)) / sigma2))
            return F

        def HighTKD2(conv_nn_module, k11, k12):
            bias = conv_nn_module.bias
            stride = conv_nn_module.stride
            padding = conv_nn_module.padding
            conv = conv_nn_module.weight.detach().numpy()  # [K2, K1, kernel size, kernel size]
            conv = conv.transpose([2, 3, 1, 0])  # [kernel size, kernel size, K1, K2]
            dim_tensor = conv.shape  # [kernel size, kernel size, K1, K2]
            conv = conv.reshape([3, 3, k11, k12, dim_tensor[3]])  # [kernel size, kernel size, k11, k12, K2]

            conv_k11 = conv.transpose([2, 0, 1, 3, 4])  # [k11, kernel size, kernel size, k12, K2]
            conv_k11 = conv_k11.reshape([k11, -1])  # [k11, D*D*k12*k2]
            cacb = 10
            sigma2 = 0.004
            start = time.clock()
            u3, s3, vt3 = VBMF(conv_k11, cacb, sigma2)
            end = time.clock()
            r31 = u3.shape[1]
            print('The rank of mode-k11: {}'.format(r31))
            print('The time of VBMF: {}'.format(end - start))
            U3 = u3  # [k11, r31]

            conv_k12 = conv.transpose([3, 0, 1, 2, 4])  # [k12, kernel size, kernel size, k11, K2]
            conv_k12 = conv_k12.reshape([k12, -1])
            start = time.clock()
            u4, s4, vt4 = VBMF(conv_k12)
            end = time.clock()
            r32 = u4.shape[1]
            print('The rank of mode-k12: {}'.format(r32))
            print('The time of VBMF: {}'.format(end - start))
            U4 = u4  # [k12, r32]

            conv_k2 = conv.transpose([4, 0, 1, 2, 3])  # [K2, kernel size, kernel size, k11, k12]
            conv_k2 = conv_k2.reshape([dim_tensor[3], -1])
            start = time.clock()
            u5, s5, vt5 = VBMF(conv_k2)
            end = time.clock()
            r4 = u4.shape[1]
            print('The rank of mode-k2: {}'.format(r4))
            print('The time of VBMF: {}'.format(r4))
            U5 = u5  # [k2, r4]

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
            print("r31, r32, r4: {}, {}, {}.".format(r31, r32, r4))
            return conv_k11, conv_k12, conv_k2, conv_c, bias, stride, padding

        super(HighTkd2ConvVbmf, self).__init__()
        conv_k11, conv_k12, conv_k2, conv_c, bias, stride, padding = HighTKD2(conv_nn_module, k11, k12)
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
    x = torch.rand([10, 256, 5, 5])
    model = torchvision.models.vgg16(pretrained=True)
    print(model)
    conv_nn_module = model.features[12]
    highertkd_conv = HighTkd2ConvVbmf(conv_nn_module, k11, k12)
    out = conv_nn_module(x)
    print(out.shape)
    out2 = highertkd_conv(x)
    print(out2.shape)


if __name__ == '__main__':
    test()