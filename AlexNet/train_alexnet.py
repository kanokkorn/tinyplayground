import os
import pickle
import tarfile
import requests
import io
import numpy as np


from PIL import Image

from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict


CKPT_PATH = "checkpoint.safetensors"

class AlexNet:
  def __init__(self, num_classes=10):
    self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
    self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
    self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(256 * 6 * 6, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, num_classes)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x).relu()
    x = x.max_pool2d(kernel_size=3, stride=2)
    x = self.conv2(x).relu()
    x = x.max_pool2d(kernel_size=3, stride=2)
    x = self.conv3(x).relu()
    x = self.conv4(x).relu()
    x = self.conv5(x).relu()
    x = x.max_pool2d(kernel_size=3, stride=2)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x).relu().dropout(0.5)
    x = self.fc2(x).relu().dropout(0.5)
    x = self.fc3(x)
    return x

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

if __name__ == "__main__":
  model = AlexNet()
  opt = SGD(get_parameters(model), lr=0.01)

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
