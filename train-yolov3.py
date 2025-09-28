import sys
import io
import time
import math
import numpy as np

from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2d, Conv2d
from tinygrad.helpers import fetch


# stole from yolov3.py
class Darknet:
  def __init__(self, cfg):
    self.blocks = parse_cfg(cfg)
    self.net_info, self.module_list = self.create_modules(self.blocks)
    print("Modules length:", len(self.module_list))

  def create_modules(self, blocks):
    net_info = blocks[0] # Info about model hyperparameters
    prev_filters, filters = 3, None
    output_filters, module_list = [], []
    ## module
    for index, x in enumerate(blocks[1:]):
      module_type = x["type"]
      module = []
      if module_type == "convolutional":
        try:
          batch_normalize, bias = int(x["batch_normalize"]), False
        except:
          batch_normalize, bias = 0, True
        # layer
        activation = x["activation"]
        filters = int(x["filters"])
        padding = int(x["pad"])
        pad = (int(x["size"]) - 1) // 2 if padding else 0
        module.append(Conv2d(prev_filters, filters, int(x["size"]), int(x["stride"]), pad, bias=bias))
        # BatchNorm2d
        if batch_normalize:
          module.append(BatchNorm2d(filters, eps=1e-05, track_running_stats=True))
        # LeakyReLU activation
        if activation == "leaky":
          module.append(lambda x: x.leaky_relu(0.1))
      elif module_type == "maxpool":
        size, stride = int(x["size"]), int(x["stride"])
        module.append(lambda x: x.max_pool2d(kernel_size=(size, size), stride=stride))
      elif module_type == "upsample":
        module.append(lambda x: Tensor(x.numpy().repeat(2, axis=-2).repeat(2, axis=-1)))
      elif module_type == "route":
        x["layers"] = x["layers"].split(",")
        # Start of route
        start = int(x["layers"][0])
        # End if it exists
        try:
          end = int(x["layers"][1])
        except:
          end = 0
        if start > 0: start -= index
        if end > 0: end -= index
        module.append(lambda x: x)
        if end < 0:
          filters = output_filters[index + start] + output_filters[index + end]
        else:
          filters = output_filters[index + start]
      # Shortcut corresponds to skip connection
      elif module_type == "shortcut":
        module.append(lambda x: x)
      elif module_type == "yolo":
        mask = list(map(int, x["mask"].split(",")))
        anchors = [int(a) for a in x["anchors"].split(",")]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        module.append([anchors[i] for i in mask])
      # Append to module_list
      module_list.append(module)
      if filters is not None:
        prev_filters = filters
      output_filters.append(filters)
    return (net_info, module_list)

  def dump_weights(self):
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]
      if module_type == "convolutional":
        print(self.blocks[i + 1]["type"], "weights", i)
        model = self.module_list[i]
        conv = model[0]
        print(conv.weight.numpy()[0][0][0])
        if conv.bias is not None:
          print("biases")
          print(conv.bias.shape)
          print(conv.bias.numpy()[0][0:5])
        else:
          print("None biases for layer", i)

  def load_weights(self, url):
    weights = np.frombuffer(fetch(url).read_bytes(), dtype=np.float32)[5:]
    ptr = 0
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]
      if module_type == "convolutional":
        model = self.module_list[i]
        try: # we have batchnorm, load conv weights without biases, and batchnorm values
          batch_normalize = int(self.blocks[i+1]["batch_normalize"])
        except: # no batchnorm, load conv weights + biases
          batch_normalize = 0
        conv = model[0]
        if batch_normalize:
          bn = model[1]
          # Get the number of weights of batchnorm
          num_bn_biases = math.prod(bn.bias.shape)
          # Load weights
          bn_biases = Tensor(weights[ptr:ptr + num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          bn_weights = Tensor(weights[ptr:ptr+num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          bn_running_mean = Tensor(weights[ptr:ptr+num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          bn_running_var = Tensor(weights[ptr:ptr+num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          # Cast the loaded weights into dims of model weights
          bn_biases = bn_biases.reshape(shape=tuple(bn.bias.shape))
          bn_weights = bn_weights.reshape(shape=tuple(bn.weight.shape))
          bn_running_mean = bn_running_mean.reshape(shape=tuple(bn.running_mean.shape))
          bn_running_var = bn_running_var.reshape(shape=tuple(bn.running_var.shape))
          # Copy data
          bn.bias = bn_biases
          bn.weight = bn_weights
          bn.running_mean = bn_running_mean
          bn.running_var = bn_running_var
        else:
          # load biases of the conv layer
          num_biases = math.prod(conv.bias.shape)
          # Load weights
          conv_biases = Tensor(weights[ptr: ptr+num_biases].astype(np.float32))
          ptr += num_biases
          # Reshape
          conv_biases = conv_biases.reshape(shape=tuple(conv.bias.shape))
          # Copy
          conv.bias = conv_biases
        # Load weighys for conv layers
        num_weights = math.prod(conv.weight.shape)
        conv_weights = Tensor(weights[ptr:ptr+num_weights].astype(np.float32))
        ptr += num_weights
        conv_weights = conv_weights.reshape(shape=tuple(conv.weight.shape))
        conv.weight = conv_weights

  def forward(self, x):
    modules = self.blocks[1:]
    outputs = {} # Cached outputs for route layer
    detections, write = None, False
    for i, module in enumerate(modules):
      module_type = (module["type"])
      if module_type == "convolutional" or module_type == "upsample":
        for layer in self.module_list[i]:
          x = layer(x)
      elif module_type == "route":
        layers = module["layers"]
        layers = [int(a) for a in layers]
        if (layers[0]) > 0:
          layers[0] = layers[0] - i
        if len(layers) == 1:
          x = outputs[i + (layers[0])]
        else:
          if (layers[1]) > 0: layers[1] = layers[1] - i
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]
          x = Tensor(np.concatenate((map1.numpy(), map2.numpy()), axis=1))
      elif module_type == "shortcut":
        from_ = int(module["from"])
        x = outputs[i - 1] + outputs[i + from_]
      elif module_type == "yolo":
        anchors = self.module_list[i][0]
        inp_dim = int(self.net_info["height"])  # 416
        num_classes = int(module["classes"])
        x = predict_transform(x, inp_dim, anchors, num_classes)
        if not write:
          detections, write = x, True
        else:
          detections = Tensor(np.concatenate((detections.numpy(), x.numpy()), axis=1))
      outputs[i] = x
    return detections

if __name__ == "__main__":
  main()
