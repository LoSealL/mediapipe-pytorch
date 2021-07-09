import torch
import torch.nn as nn

from .blaze_block import BlazeBlock, ConvTF


class BackBone1(nn.Module):
    def __init__(self):
        super(BackBone1, self).__init__()
        self.backbone1 = nn.ModuleList([
            ConvTF(3, 32, 5, 2),
            nn.PReLU(32),
            BlazeBlock(32, 32, 5),
            nn.PReLU(32),
            BlazeBlock(32, 32, 5),
            nn.PReLU(32),
            BlazeBlock(32, 32, 5),
            nn.PReLU(32),
            BlazeBlock(32, 64, 5, 2),
            nn.PReLU(64),
            BlazeBlock(64, 64, 5, 1),
            nn.PReLU(64),
            BlazeBlock(64, 64, 5, 1),
            nn.PReLU(64),
            BlazeBlock(64, 64, 5, 1),
            nn.PReLU(64),
            BlazeBlock(64, 128, 5, 2),
            nn.PReLU(128),
            BlazeBlock(128, 128, 5, 1),
            nn.PReLU(128),
            BlazeBlock(128, 128, 5, 1),
            nn.PReLU(128),
            BlazeBlock(128, 128, 5, 1),
            nn.PReLU(128)
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone1:
            y = fn(y)
        return y


class BackBone2(nn.Module):
    def __init__(self):
        super(BackBone2, self).__init__()
        self.path1 = nn.Sequential(
            BlazeBlock(128, 256, 5, 2),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256))
        self.path2 = nn.Sequential(
            BlazeBlock(256, 256, 5, 2),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvTF(256, 256, 1),
            nn.PReLU(256))
        self.path3 = nn.Sequential(
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256),
            BlazeBlock(256, 256, 5, 1),
            nn.PReLU(256))
        self.path4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvTF(256, 128, 1),
            nn.PReLU(128))
        self.path5 = nn.Sequential(
            BlazeBlock(128, 128, 5),
            nn.PReLU(128),
            BlazeBlock(128, 128, 5),
            nn.PReLU(128))

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(p1)
        p3 = self.path3(p1 + p2)
        p4 = self.path4(p3)
        p5 = self.path5(x + p4)
        return p3, p5


class BlazePalm(nn.Module):
    IMAGE_SHAPE = [3, 128, 128]
    KEY_POINTS_NUMBER = 7
    NUM_PER_KEYPOINT = 2
    NUM_PER_BOX = 4

    def __init__(self):
        super(BlazePalm, self).__init__()
        self.backbone1 = BackBone1()
        self.backbone2 = BackBone2()
        self.classifier2 = nn.Conv2d(128, 2, 1)
        self.classifier1 = nn.Conv2d(256, 6, 1)
        self.regressor2 = nn.Conv2d(128, 36, 1)
        self.regressor1 = nn.Conv2d(256, 108, 1)

    def forward(self, image):
        b1 = self.backbone1(image)
        f1, f2 = self.backbone2(b1)
        c1 = self.classifier1(f1)
        c2 = self.classifier2(f2)
        r1 = self.regressor1(f1)
        r2 = self.regressor2(f2)

        regression_channels = self.NUM_PER_BOX + self.KEY_POINTS_NUMBER * self.NUM_PER_KEYPOINT
        c1 = c1.permute(0, 2, 3, 1).reshape(-1, 384, 1)
        c2 = c2.permute(0, 2, 3, 1).reshape(-1, 512, 1)
        r1 = r1.permute(0, 2, 3, 1).reshape(-1, 384, regression_channels)
        r2 = r2.permute(0, 2, 3, 1).reshape(-1, 512, regression_channels)

        c = torch.cat((c2, c1), dim=1)
        r = torch.cat((r2, r1), dim=1)
        return c, r
