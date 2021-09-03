import torch
import torch.nn as nn

from .facebox_decoder import FaceBoxesDecode
from .faceboxes import FaceBoxes
from .prior_box import PriorBox

FACEBOXES_OPTIONS = {
    'image_width': 1024,
    'image_height': 1024,
    'min_size_stride': [[32, 16, 8], [4], [2]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}


class FaceBoxesDetection(nn.Module):
    IMAGE_NORMALIZE = False
    NUM_CLASSES = 2
    NUM_COORDS = 4

    def __init__(self, image_scale=1, image_size=None):
        super().__init__()
        option = FACEBOXES_OPTIONS.copy()
        option['image_width'] *= image_scale
        option['image_height'] *= image_scale
        if image_size:
            option['image_width'] = image_size[1]
            option['image_height'] = image_size[0]
        anchors = PriorBox(option).forward()
        self.faceboxes = FaceBoxes(phase='test', size=image_size, num_classes=self.NUM_CLASSES)
        self.decode = FaceBoxesDecode(self.NUM_CLASSES, self.NUM_COORDS, anchors, option['variance'])
        if self.IMAGE_NORMALIZE:
            scale = torch.tensor([104, 117, 123], requires_grad=False).reshape([1, 3, 1, 1])
            self.register_buffer('scale', scale)

    def forward(self, image: torch.Tensor):
        if self.IMAGE_NORMALIZE:
            image = image.float()
            image -= self.scale
        raw_box, raw_score = self.faceboxes(image)
        score, box = self.decode(raw_score, raw_box)
        return score, box

    def load_state_dict(self, state_dict, strict: bool=False):
        return self.faceboxes.load_state_dict(state_dict, strict)
