from tinygrad.nn.state import safe_load
import torch
import torch.nn as nn
import torch.onnx

class PyTorchAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, 4, 2)
        self.conv2 = nn.Conv2d(64, 192, 5, 1, 2)
        self.conv3 = nn.Conv2d(192, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 3, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 3, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.max_pool2d(x, 3, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def convert_safetensors_to_onnx(ckpt_path="checkpoint.safetensors", onnx_path="model.onnx"):
    state = safe_load(ckpt_path)
    model = PyTorchAlexNet()
    
    # Load weights (map tinygrad names to PyTorch)
    model.conv1.weight.data = torch.from_numpy(state["conv1.weight"].numpy())
    model.conv1.bias.data = torch.from_numpy(state["conv1.bias"].numpy())
    # ... repeat for conv2–conv5, fc1–fc3 (add all mappings)
    
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, onnx_path, opset_version=17,
                      input_names=["input"], output_names=["output"])
    print(f"✓ Exported to {onnx_path}")

if __name__ == "__main__":
    convert_safetensors_to_onnx()
