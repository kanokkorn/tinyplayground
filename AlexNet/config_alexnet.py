ALEXNET_CONFIG = [
    {"name": "conv1", "type": "conv", "in": 3, "out": 64,  "k": 11, "s": 4, "p": 2},
    {"name": "pool1", "type": "maxpool", "k": 3, "s": 2},
    {"name": "conv2", "type": "conv", "in": 64, "out": 192, "k": 5, "s": 1, "p": 2},
    {"name": "pool2", "type": "maxpool", "k": 3, "s": 2},
    {"name": "conv3", "type": "conv", "in": 192, "out": 384, "k": 3, "s": 1, "p": 1},
    {"name": "conv4", "type": "conv", "in": 384, "out": 256, "k": 3, "s": 1, "p": 1},
    {"name": "conv5", "type": "conv", "in": 256, "out": 256, "k": 3, "s": 1, "p": 1},
    {"name": "pool3", "type": "maxpool", "k": 3, "s": 2},
    {"name": "flatten", "type": "flatten"},
    {"name": "fc1", "type": "linear", "in": 256*6*6, "out": 4096},
    {"name": "fc2", "type": "linear", "in": 4096, "out": 4096},
    {"name": "fc3", "type": "linear", "in": 4096, "out": 10},
]
