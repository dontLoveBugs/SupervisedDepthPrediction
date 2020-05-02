#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 16:46
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : layers.py
"""




class FullImageEncoder(nn.Module):
    def __init__(self):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(16, stride=16, padding=(8, 8))  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * 4 * 5, 512)  # kitti 4x5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        self.upsample = nn.UpsamplingBilinear2d(size=(49, 65))  # KITTI 49X65 NYU 33X45

    def forward(self, x):
        # print('x size:', x.size())
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 4 * 5)  # kitti 4x5
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = self.upsample(x5)
        return out


class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super(SceneUnderstandingModule, self).__init__()
        self.encoder = FullImageEncoder()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 142, 1),  # KITTI 142 NYU 136 In paper, K = 80 is best, so use 160 is good!
            nn.UpsamplingBilinear2d(size=(385, 513))
        )

    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        return out


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1)
        C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W)
        return decode_c, ord_c1