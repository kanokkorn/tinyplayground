import os
import io 
import pickle
import tarfile
import requests

import numpy as np
from PIL import Image

from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict
from config_alexnet import ALEXNET_CONFIG

CKPT_PATH = "checkpoint.safetensors"

def build_tinygrad_model(config):
  layers = []
  for layer in config:
    if layer["type"] == "conv":
      layers.append(nn.Conv2d(layer["in"], layer["out"], layer["k"], 
            stride=layer["s"], padding=layer["p"]))
    elif layer["type"] == "maxpool":
      layers.append(("maxpool", layer["k"], layer["s"]))
    elif layer["type"] == "linear":
      layers.append(nn.Linear(layer["in"], layer["out"]))
    elif layer["type"] == "flatten":
      layers.append("flatten")
    return layers

def get_batch(batch_size=4):
  data_dir = "dataset/train"
  if os.path.exists(data_dir):
    images, labels = [], []
    classes = sorted(os.listdir(data_dir))
    for i, cls in enumerate(classes):
      for f in os.listdir(os.path.join(data_dir, cls))[:batch_size//len(classes)]:
        img = Image.open(os.path.join(data_dir, cls, f)).convert("RGB").resize((224, 224))
        images.append(np.array(img).transpose(2, 0, 1) / 255.0)
        labels.append(i)
    return Tensor(np.stack(images)).float(), Tensor(labels)
  else:
    if not os.path.exists("data/cifar-10-batches-py"):
      print("Downloading CIFAR-10...")
      r = requests.get("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", stream=True)
      with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
          tar.extractall("data")
    with open("data/cifar-10-batches-py/data_batch_1", "rb") as f:
      data = pickle.load(f, encoding="bytes")
    imgs = data[b"data"].reshape(-1, 3, 32, 32)[:batch_size]
    imgs = np.array([np.array(Image.fromarray(img.transpose(1,2,0)).resize((224,224))).transpose(2,0,1) for img in imgs]) / 255.0
  return Tensor(imgs).float(), Tensor(data[b"labels"][:batch_size])

def save_checkpoint(model, opt, path=CKPT_PATH):
  state = get_state_dict(model)
  state.update({f"opt_{k}": v for k, v in get_state_dict(opt).items()})
  safe_save(state, path)
  print(f"✓ Saved checkpoint to {path}")

def load_checkpoint(model, opt, path=CKPT_PATH):
  if os.path.exists(path):
    state = safe_load(path)
    load_state_dict(model, {k: v for k, v in state.items() if not k.startswith("opt_")})
    load_state_dict(opt, {k[4:]: v for k, v in state.items() if k.startswith("opt_")})
    print(f"✓ Resumed from {path}")
    return True
  return False

class AlexNet:
  def __init__(self):
    self.layers = build_tinygrad_model(ALEXNET_CONFIG)

  def __call__(self, x):
    for layer in self.layers:
      if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        x = layer(x)
      elif isinstance(layer, tuple) and layer[0] == "maxpool":
        x = x.max_pool2d(kernel_size=layer[1], stride=layer[2])
      elif layer == "flatten":
        x = x.reshape(x.shape[0], -1)
        x = x.relu() if not isinstance(layer, tuple) else x
    return x

if __name__ == "__main__":
  model = AlexNet()
  opt = SGD(get_parameters(model), lr=0.03)

  load_checkpoint(model, opt)

  Tensor.training = True
  for epoch in range(50):
    x, y = get_batch()
    out = model(x)
    loss = out.sparse_categorical_crossentropy(y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Epoch {epoch+1} | Loss: {loss.numpy():.4f}")

    save_checkpoint(model, opt)   # autosave every batch

  Tensor.training = False
