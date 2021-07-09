import torch
import torch.nn as nn

from .blaze_block import ConvTF, BlazeBlock


class BackBone1(nn.Module):
    def __init__(self):
        super(BackBone1, self).__init__()
        self.backbone1 = nn.ModuleList([
            ConvTF(3, 24, 5, 2),
            nn.ReLU(True),
            BlazeBlock(24, 24, 3),
            nn.ReLU(True),
            BlazeBlock(24, 28, 3),
            nn.ReLU(True),
            BlazeBlock(28, 32, 3, 2),
            nn.ReLU(True),
            BlazeBlock(32, 36, 3),
            nn.ReLU(True),
            BlazeBlock(36, 42, 3),
            nn.ReLU(True),
            BlazeBlock(42, 48, 3, 2),
            nn.ReLU(True),
            BlazeBlock(48, 56, 3),
            nn.ReLU(True),
            BlazeBlock(56, 64, 3),
            nn.ReLU(True),
            BlazeBlock(64, 72, 3),
            nn.ReLU(True),
            BlazeBlock(72, 80, 3),
            nn.ReLU(True),
            BlazeBlock(80, 88, 3),
            nn.ReLU(True)
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone1:
            y = fn(y)
        return y


class BackBone2(nn.Module):
    def __init__(self):
        super(BackBone2, self).__init__()
        self.backbone2 = nn.ModuleList([
            BlazeBlock(88, 96, 3, 2),
            nn.ReLU(),
            BlazeBlock(96, 96, 3),
            nn.ReLU(True),
            BlazeBlock(96, 96, 3),
            nn.ReLU(True),
            BlazeBlock(96, 96, 3),
            nn.ReLU(True),
            BlazeBlock(96, 96, 3),
            nn.ReLU(True)
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone2:
            y = fn(y)
        return y


class BlazeFace(nn.Module):
    IMAGE_SHAPE = [3, 128, 128]
    KEY_POINTS_NUMBER = 6
    NUM_PER_KEYPOINT = 2
    NUM_PER_BOX = 4

    def __init__(self):
        super(BlazeFace, self).__init__()
        self.backbone1 = BackBone1()
        self.backbone2 = BackBone2()
        self.classifier1 = nn.Conv2d(88, 2, 1)
        self.classifier2 = nn.Conv2d(96, 6, 1)
        self.regressor1 = nn.Conv2d(88, 32, 1)
        self.regressor2 = nn.Conv2d(96, 96, 1)

    def forward(self, image):
        b1 = self.backbone1(image)
        b2 = self.backbone2(b1)
        c1 = self.classifier1(b1)
        c2 = self.classifier2(b2)
        r1 = self.regressor1(b1)
        r2 = self.regressor2(b2)

        regression_channels = self.NUM_PER_BOX + self.KEY_POINTS_NUMBER * self.NUM_PER_KEYPOINT
        c1 = c1.permute(0, 2, 3, 1).reshape(-1, 512, 1)
        c2 = c2.permute(0, 2, 3, 1).reshape(-1, 384, 1)
        r1 = r1.permute(0, 2, 3, 1).reshape(-1, 512, regression_channels)
        r2 = r2.permute(0, 2, 3, 1).reshape(-1, 384, regression_channels)

        c = torch.cat((c1, c2), dim=1)
        r = torch.cat((r1, r2), dim=1)
        return c, r
