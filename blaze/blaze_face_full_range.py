import torch
import torch.nn as nn

from .blaze_block import ConvTF, DualBlazeBlock


class BackBone1(nn.Module):
    def __init__(self):
        super(BackBone1, self).__init__()
        self.backbone1 = nn.ModuleList([
            ConvTF(3, 32, 3, 2),  # 96x96x32
            nn.ReLU(True),
            DualBlazeBlock(32, 8, 32, 3),
            nn.ReLU(True),
            DualBlazeBlock(32, 8, 32, 3),
            nn.ReLU(True),
            DualBlazeBlock(32, 12, 48, 3, 2),  # 48x48x48
            nn.ReLU(True),
            DualBlazeBlock(48, 12, 48, 3),
            nn.ReLU(True),
            DualBlazeBlock(48, 16, 64, 3),  # 48x48x64
            nn.ReLU(True),
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone1:
            y = fn(y)
        return y


class BackBone2(nn.Module):
    def __init__(self):
        super(BackBone2, self).__init__()
        # 48x48x64
        self.backbone2 = nn.ModuleList([
            DualBlazeBlock(64, 24, 96, 3, 2),  # 24x24x96
            nn.ReLU(True),
            DualBlazeBlock(96, 32, 128, 3),
            nn.ReLU(True),
            DualBlazeBlock(128, 32, 128, 3),
            nn.ReLU(True),
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone2:
            y = fn(y)
        return y


class BackBone3(nn.Module):
    def __init__(self):
        super(BackBone3, self).__init__()
        # 24x24x128
        self.backbone3 = nn.ModuleList([
            DualBlazeBlock(128, 36, 144, 3, 2),  # 12x12x144
            nn.ReLU(True),
            DualBlazeBlock(144, 48, 192, 3),
            nn.ReLU(True),
            DualBlazeBlock(192, 64, 256, 3),  # 12x12x256
            nn.ReLU(True),
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone3:
            y = fn(y)
        return y


class BackBone4(nn.Module):
    def __init__(self):
        super(BackBone4, self).__init__()
        # 12x12x256
        self.backbone4 = nn.ModuleList([
            DualBlazeBlock(256, 96, 384, 3, 2),  # 6x6x384
            nn.ReLU(True),
            DualBlazeBlock(384, 96, 384, 3),
            nn.ReLU(True),
            DualBlazeBlock(384, 96, 384, 3),
            nn.ReLU(True),
            DualBlazeBlock(384, 96, 384, 3),
            nn.ReLU(True),
            ConvTF(384, 96, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 12x12x96
        ])
        self.bridge3 = nn.Sequential(
            ConvTF(256, 96, 1),
            nn.ReLU(True)
        )
        self.tail3 = nn.ModuleList([
            DualBlazeBlock(96, 48, 96, 3),
            nn.ReLU(True),
            DualBlazeBlock(96, 48, 96, 3),
            nn.ReLU(True),
            ConvTF(96, 64, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 24x24x64
        ])
        self.bridge2 = nn.Sequential(
            ConvTF(128, 64, 1),
            nn.ReLU(True)
        )
        self.tail2 = nn.ModuleList([
            DualBlazeBlock(64, 32, 64, 3),
            nn.ReLU(True),
            DualBlazeBlock(64, 32, 64, 3),
            nn.ReLU(True),
            ConvTF(64, 48, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 48x48x48
        ])
        self.bridge1 = nn.Sequential(
            ConvTF(64, 48, 1),
            nn.ReLU(True)
        )
        self.tail1 = nn.ModuleList([
            DualBlazeBlock(48, 24, 48, 3),
            nn.ReLU(True)  # 48x48x48
        ])

    def forward(self, b3, b2, b1):
        y = b3
        for fn in self.backbone4:
            y = fn(y)
        y += self.bridge3(b3)
        for fn in self.tail3:
            y = fn(y)
        y += self.bridge2(b2)
        for fn in self.tail2:
            y = fn(y)
        y += self.bridge1(b1)
        for fn in self.tail1:
            y = fn(y)
        return y


class BlazeFaceFR(nn.Module):
    IMAGE_SHAPE = [3, 192, 192]
    KEY_POINTS_NUMBER = 6
    NUM_PER_KEYPOINT = 2
    NUM_PER_BOX = 4

    def __init__(self, image_scale=(1, 1)):
        super().__init__()
        self.backbone1 = BackBone1()
        self.backbone2 = BackBone2()
        self.backbone3 = BackBone3()
        self.backbone4 = BackBone4()
        self.classifier1 = nn.Conv2d(48, 1, 1)
        self.regressor1 = nn.Conv2d(48, 16, 1)
        self.scale = image_scale[0] * image_scale[1]

    def forward(self, image):
        b1 = self.backbone1(image)
        b2 = self.backbone2(b1)
        b3 = self.backbone3(b2)
        b4 = self.backbone4(b3, b2, b1)
        c1 = self.classifier1(b4)
        r1 = self.regressor1(b4)

        regression_channels = self.NUM_PER_BOX + self.KEY_POINTS_NUMBER * self.NUM_PER_KEYPOINT
        c1 = c1.permute(0, 2, 3, 1).reshape(-1, int(2304 * self.scale), 1)
        r1 = r1.permute(0, 2, 3, 1).reshape(-1, int(2304 * self.scale), regression_channels)

        return c1, r1
