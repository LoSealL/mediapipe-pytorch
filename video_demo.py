import argparse

import cv2 as cv
import torch
import torchvision as tv
import tqdm
from PIL import Image

from detect_result import DetectResult

parser = argparse.ArgumentParser()
parser.add_argument("video", help="a video file")
parser.add_argument("--model", default='blaze', choices=('blaze', 'blaze_full_range', 'faceboxes'))
parser.add_argument("--weight", default='model/face_detection_short_range.pth')
parser.add_argument("--scale", type=float, default=1.0, help="scale the model input size, exclusive to --size")
parser.add_argument("--size", nargs=2, default=None, type=int, help="model input size, exclusive to --scale")
parser.add_argument("--output_dir", default=None, help="save visualized image")
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--show", action='store_true', help="show a preview window")
args = parser.parse_args()

if args.model == 'blaze':
    # This is for mediapipe blazeface short range
    from blaze.face_detection import FaceDetection

    FaceDetection.IMAGE_NORMALIZE = True
    model = FaceDetection(args.scale, args.size)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    size = torch.tensor([128, 128]) * args.scale
    filter = DetectResult(FaceDetection.NUM_CLASSES, FaceDetection.NUM_COORDS, 0.75, 0.3)
elif args.model == 'blaze_full_range':
    # This is for mediapipe blazeface full range
    from blaze.face_detection import FaceDetectionFullRange

    FaceDetectionFullRange.IMAGE_NORMALIZE = True
    model = FaceDetectionFullRange(args.scale, args.size)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    size = torch.tensor([192, 192]) * args.scale
    filter = DetectResult(FaceDetectionFullRange.NUM_CLASSES, FaceDetectionFullRange.NUM_COORDS, 0.6, 0.3)
elif args.model == 'faceboxes':
    from others.faceboxes import FaceBoxesDetection

    FaceBoxesDetection.IMAGE_NORMALIZE = True
    if args.size is None:
        size = [1024 * args.scale, 1024 * args.scale]
    else:
        size = args.size
    model = FaceBoxesDetection(args.scale, size)
    model.load_state_dict(torch.load(args.weight))
    filter = DetectResult(1, 4, 0.5, 0.3)
    size = torch.tensor(size)

model.eval()
if args.cuda:
    model = model.cuda()
    filter = filter.cuda()
if args.size:
    size = torch.tensor(args.size)
size = size.long().cpu().numpy().tolist()
vframes, _, info = tv.io.read_video(args.video)
for i, frame in enumerate(tqdm.tqdm(vframes)):
    h, w, c = frame.shape
    sy, sx = size[0] / h, size[1] / w
    right = int(size[1] / min(sy, sx) - w)
    bottom = int(size[0] / min(sy, sx) - h)
    frame = frame.permute([2, 0, 1])
    trans = tv.transforms.Compose([
        tv.transforms.Pad([0, 0, right, bottom], fill=0),
        tv.transforms.Resize(size=size)
    ])
    image = trans(frame)[None]
    if args.cuda:
        image = image.cuda()
    _score, _box = model(image)
    scores, boxes = filter(_score, _box)
    scores = scores.cpu().detach().numpy()
    boxes = boxes.cpu().detach().numpy()

    if args.output_dir or args.show:
        img = frame.detach().cpu().numpy().transpose([1, 2, 0])
        for box in boxes:
            box, points = box[:4].reshape([2, 2]), box[4:].reshape([-1, 2])
            lt = ((box[0] - box[1] / 2) * [w + right, h + bottom]).astype('int32')
            rb = ((box[0] + box[1] / 2) * [w + right, h + bottom]).astype('int32')
            img = cv.rectangle(img, lt, rb, color=(242, 159, 16), thickness=2)
        if args.output_dir:
            Image.fromarray(img, 'RGB').save(f'./{args.output_dir}/frame_{i:04d}.png')
        elif args.show:
            cv.imshow("demo", img[..., ::-1])
            # press ESC to exit
            if cv.waitKey(1) == 27:
                break
