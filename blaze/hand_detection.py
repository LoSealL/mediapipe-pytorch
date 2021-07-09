import torch

from .blaze_palm import BlazePalm
from .blaze_decoder import generate_anchors, BlazeDecodeBox

import torch.nn as nn

# options from `mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt`
SSD_ANCHOR_OPTIONS = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}


class HandDetection(nn.Module):
    IMAGE_NORMALIZE = False
    NO_DECODE = False
    NUM_CLASSES = 1
    NUM_BOXES = 896
    NUM_COORDS = 18
    X_SCALE = 128.0
    Y_SCALE = 128.0
    W_SCALE = 128.0
    H_SCALE = 128.0

    def __init__(self):
        super(HandDetection, self).__init__()
        anchors = generate_anchors(SSD_ANCHOR_OPTIONS)
        anchors = torch.tensor(anchors, requires_grad=False)
        self.blazepalm = BlazePalm()
        self.decode = BlazeDecodeBox(self.NUM_CLASSES, self.NUM_BOXES, self.NUM_COORDS, self.X_SCALE, self.Y_SCALE,
                                     self.W_SCALE, self.H_SCALE, anchors)

    def forward(self, image: torch.Tensor):
        # normalize to [-1, 1]
        if self.IMAGE_NORMALIZE:
            image = image.float() / 127.5 - 1.0
        raw_score, raw_box = self.blazepalm(image)
        if self.NO_DECODE:
            return raw_score, raw_box
        score, box = self.decode(raw_score, raw_box)
        return score, box
