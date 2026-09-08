from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict
import os, pickle, tarfile, requests, io, numpy as np
from PIL import Image

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

  # --- TinyJit Step Functions ---
  @TinyJit
  def train_step(x: Tensor, y: Tensor) -> Tensor:
    opt.zero_grad()
    out = model(x)
    loss = out.sparse_categorical_crossentropy(y)
    loss.backward()
    opt.step()
    return loss.realize()

  @TinyJit
  def eval_step(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    out = model(x)
    loss = out.sparse_categorical_crossentropy(y)
    correct = (out.argmax(axis=-1) == y)
    return loss.realize(), correct.realize()

  @TinyJit
  def infer_step(x: Tensor) -> Tensor:
    return model(x).softmax().realize()

  # --- Helper Functions ---
  def evaluate(steps: int = 5, batch_size: int = 4):
    Tensor.training = False
    total_loss, total_correct = 0.0, 0
    for _ in range(steps):
      x, y = get_batch(batch_size=batch_size)
      loss, correct = eval_step(x, y)
      total_loss += loss.item()
      total_correct += correct.sum().item()
    avg_loss = total_loss / steps
    accuracy = total_correct / (steps * batch_size)
    print(f"[EVAL] Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy

  def predict(image_path: str):
    if not os.path.exists(image_path):
      print(f"File not found: {image_path}")
      return
    Tensor.training = False
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_arr = np.array(img).transpose(2, 0, 1) / 255.0
    x = Tensor(np.expand_dims(img_arr, axis=0)).float() # batch_size = 1

    probs = infer_step(x)
    pred_class = probs.argmax(axis=-1).item()
    confidence = probs[0, pred_class].item()
    print(f"[INFER] Class: {pred_class} | Confidence: {confidence * 100:.2f}%")
    return pred_class, confidence

  # --- Training Loop ---
  Tensor.training = True
  for epoch in range(50):
    x, y = get_batch(batch_size=4)
    loss = train_step(x, y)
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
      save_checkpoint(model, opt)
      evaluate(steps=5, batch_size=4)
      Tensor.training = True # Switch back for training steps

  # --- Final Evaluation ---
  print("\nFinal Model Evaluation:")
  evaluate(steps=10, batch_size=4)
