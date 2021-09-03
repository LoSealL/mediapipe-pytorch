"""
Copyright: Wenyi Tang 2021
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2021/8/9

Show complexity
"""
import torch

from blaze.face_detection import FaceDetection, FaceDetectionFullRange
from blaze.hand_detection import HandDetection
from blaze.hand_landmark import HandLandmark
from others.faceboxes import FaceBoxesDetection
from ptflop import get_model_complexity_info

fd = FaceDetection()
state = torch.load('model/face_detection_short_range.pth', map_location='cpu')
fd.load_state_dict(state, strict=True)

macs, params = get_model_complexity_info(fd, (3, 128, 128), as_strings=False, print_per_layer_stat=False)
print(f"Blaze Face Short Range: {2 * macs / 2**30:.5f}GMACs")

fdfr = FaceDetectionFullRange()
state = torch.load('model/face_detection_full_range.pth', map_location='cpu')
fdfr.load_state_dict(state, strict=True)

macs, params = get_model_complexity_info(fdfr, (3, 192, 192), as_strings=False, print_per_layer_stat=False)
print(f"Blaze Face Full Range: {2 * macs / 2**30:.5f}GMACs")

hd = HandDetection()
state = torch.load('model/palm_detection.pth', map_location='cpu')
hd.load_state_dict(state, strict=True)

macs, params = get_model_complexity_info(hd, (3, 128, 128), as_strings=False, print_per_layer_stat=False)
print(f"Blaze Hand: {2 * macs / 2**30:.5f}GMACs")

hl = HandLandmark()
macs, params = get_model_complexity_info(
    hl, tuple(HandLandmark.IMAGE_SHAPE), as_strings=False, print_per_layer_stat=False)
print(f"Hand Landmark: {2 * macs / 2**30:.5f}GMACs")

fb = FaceBoxesDetection(image_size=(320, 320))
macs, params = get_model_complexity_info(fb, (3, 320, 320), as_strings=False, print_per_layer_stat=False)
print(f"Face-Boxes: {2 * macs / 2**30:.5f}GMACs")
