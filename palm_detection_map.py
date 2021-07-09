import json

import tensorflow as tf
import torch
from tf2onnx.tflite.Model import Model

from blaze.hand_detection import HandDetection

# open tf lite model (flatbuffer)
with open('model/palm_detection.tflite', 'rb') as fd:
    model = Model.GetRootAsModel(fd.read(), 0)

assert model.SubgraphsLength() == 1
graph = model.Subgraphs(0)

# open pytorch model
model_fd = HandDetection()
state = model_fd.state_dict()

# make a list of weights in tf lite model
k = 0
with open('model/palm_detection.map', 'w') as fd:
    weights_tflite = []
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        tensor_name = tensor.Name().decode()
        if tensor.Buffer() > 0 and 'channel_padding' not in tensor_name and 'up_sampling2d' not in tensor_name:
            print(f"[{k:03d}] {tensor_name} {tensor.ShapeAsNumpy()}", file=fd)
            k += 1
            weights_tflite.append(tensor)
# make a list of weights in pytorch model
with open('model/palm_detection.map', 'a') as fd:
    fd.writelines(["\n"])
    for i, name in enumerate(state):
        print(f"[{i:03d}] {name} {state[name].shape}", file=fd)

    assert len(weights_tflite) == len(state)

    name_map = {}
    for name, tensor_lite in zip(state, weights_tflite):
        name_map[name] = tensor_lite.Name().decode()
    fd.write("\n")
    fd.write(json.dumps(name_map, indent=2))

inter = tf.lite.Interpreter('model/palm_detection.tflite')
inter.allocate_tensors()

reverse_name_map = {v: k for k, v in name_map.items()}
for i, tensor_detail in enumerate(inter.get_tensor_details()):
    name = tensor_detail['name']
    if name in reverse_name_map:
        data = inter.get_tensor(i).astype('float32')
        if 'Kernel' in name:
            # this is a conv kernel, for normal conv, it's [o, x, y, i]
            # for group conv, it's [i / g, x, y, i]
            # in blazeface, the group conv is always depthwise conv
            if data.shape[0] == 1:
                data = data.transpose([3, 0, 1, 2])
            else:
                data = data.transpose([0, 3, 1, 2])
        elif 'Alpha' in name:
            # flatten prelu parameters
            data = data.reshape([-1])
        state[reverse_name_map[name]] = torch.from_numpy(data)

torch.save(state, "model/palm_detection.pth")
try:
    model_fd.load_state_dict(state, strict=True)
    torch.onnx.export(model_fd, (torch.randn(1, 3, 128, 128),), 'model/palm_detection.onnx',
                      output_names=['classificators', 'regressors'], input_names=['input'])
except RuntimeError as ex:
    print(f"[!] Fail to load mapped weights:\n{ex}")
