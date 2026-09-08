#!/usr/bin/env python3
import time
import math
import numpy as np
from typing import List, Tuple
from tinygrad import Tensor,Device, TinyJit, nn
from tinygrad.helpers import getenv

# ==============================================================================
# 1. Darknet-53 & YOLOv3 Model Definition
# ==============================================================================

class DarknetConv2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

    def __call__(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x)).leaky_relu(0.1)

class ResidualBlock:
    def __init__(self, channels: int):
        self.conv1 = DarknetConv2D(channels, channels // 2, kernel_size=1)
        self.conv2 = DarknetConv2D(channels // 2, channels, kernel_size=3)

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))

class Darknet53:
    def __init__(self):
        self.conv0 = DarknetConv2D(3, 32, kernel_size=3)
        self.down1 = DarknetConv2D(32, 64, kernel_size=3, stride=2)
        self.res1 = [ResidualBlock(64) for _ in range(1)]
        self.down2 = DarknetConv2D(64, 128, kernel_size=3, stride=2)
        self.res2 = [ResidualBlock(128) for _ in range(2)]
        self.down3 = DarknetConv2D(128, 256, kernel_size=3, stride=2)
        self.res3 = [ResidualBlock(256) for _ in range(8)]
        self.down4 = DarknetConv2D(256, 512, kernel_size=3, stride=2)
        self.res4 = [ResidualBlock(512) for _ in range(8)]
        self.down5 = DarknetConv2D(512, 1024, kernel_size=3, stride=2)
        self.res5 = [ResidualBlock(1024) for _ in range(4)]

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.conv0(x)
        x = self.down1(x)
        for block in self.res1: x = block(x)
        x = self.down2(x)
        for block in self.res2: x = block(x)
        x = self.down3(x)
        for block in self.res3: x = block(x)
        scale_small = x  # Stride 8
        x = self.down4(x)
        for block in self.res4: x = block(x)
        scale_medium = x # Stride 16
        x = self.down5(x)
        for block in self.res5: x = block(x)
        scale_large = x  # Stride 32
        return scale_small, scale_medium, scale_large

class YOLOv3:
    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        out_dim = 3 * (5 + num_classes)
        self.backbone = Darknet53()

        # Detection scale 1 (Large objects / 13x13)
        self.head1_conv = [
            DarknetConv2D(1024, 512, 1), DarknetConv2D(512, 1024, 3),
            DarknetConv2D(1024, 512, 1), DarknetConv2D(512, 1024, 3),
            DarknetConv2D(1024, 512, 1)
        ]
        self.head1_out = [DarknetConv2D(512, 1024, 3), nn.Conv2d(1024, out_dim, 1, bias=True)]

        # Route & Upsample 1 -> Scale 2
        self.route1_conv = DarknetConv2D(512, 256, 1)
        self.head2_conv = [
            DarknetConv2D(512 + 256, 256, 1), DarknetConv2D(256, 512, 3),
            DarknetConv2D(512, 256, 1),       DarknetConv2D(256, 512, 3),
            DarknetConv2D(512, 256, 1)
        ]
        self.head2_out = [DarknetConv2D(256, 512, 3), nn.Conv2d(512, out_dim, 1, bias=True)]

        # Route & Upsample 2 -> Scale 3
        self.route2_conv = DarknetConv2D(256, 128, 1)
        self.head3_conv = [
            DarknetConv2D(256 + 128, 128, 1), DarknetConv2D(128, 256, 3),
            DarknetConv2D(256, 128, 1),       DarknetConv2D(128, 256, 3),
            DarknetConv2D(256, 128, 1)
        ]
        self.head3_out = [DarknetConv2D(128, 256, 3), nn.Conv2d(256, out_dim, 1, bias=True)]

    def _upsample2x(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        return x.reshape(b, c, h, 1, w, 1).expand(b, c, h, 2, w, 2).reshape(b, c, h * 2, w * 2)

    def __call__(self, x: Tensor) -> List[Tensor]:
        c3, c4, c5 = self.backbone(x)

        # Head 1
        for conv in self.head1_conv: c5 = conv(c5)
        out1 = self.head1_out[1](self.head1_out[0](c5))

        # Head 2
        p5 = self.route1_conv(c5)
        p5_up = self._upsample2x(p5)
        c4 = p5_up.cat(c4, dim=1)
        for conv in self.head2_conv: c4 = conv(c4)
        out2 = self.head2_out[1](self.head2_out[0](c4))

        # Head 3
        p4 = self.route2_conv(c4)
        p4_up = self._upsample2x(p4)
        c3 = p4_up.cat(c3, dim=1)
        for conv in self.head3_conv: c3 = conv(c3)
        out3 = self.head3_out[1](self.head3_out[0](c3))

        return [out1, out2, out3]

# ==============================================================================
# 2. Darknet YOLOv3 Loss & Target Formatter
# ==============================================================================

# Anchors matching pjreddie's darknet yolov3.cfg
ANCHORS = [
    [(116, 90), (156, 198), (373, 326)], # Scale 1 (13x13)
    [(30, 61),  (62, 45),   (59, 119)],   # Scale 2 (26x26)
    [(10, 13),  (16, 30),   (33, 23)]     # Scale 3 (52x52)
]

def yolov3_loss(preds: List[Tensor], targets: List[Tensor]) -> Tensor:
    """
    Computes SSE / Logistic Loss matching pjreddie darknet.
    Returns a 0-dimensional scalar tensor.
    """
    # 1. Initialize as a true 0D scalar (shape ())
    total_loss = Tensor.zeros(())
    
    for pred, target in zip(preds, targets):
        b, c, h, w = pred.shape
        num_classes = (c // 3) - 5
        
        # Reshape to (B, 3, H, W, 5 + num_classes)
        p = pred.reshape(b, 3, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2)
        
        xy = p[..., 0:2].sigmoid()
        wh = p[..., 2:4]
        conf = p[..., 4:5].sigmoid()
        cls_prob = p[..., 5:].sigmoid()
        
        mask = target[..., 0:1]
        noobj_mask = target[..., 1:2]
        txy = target[..., 2:4]
        twh = target[..., 4:6]
        tconf = target[..., 6:7]
        tcls = target[..., 7:]
        
        # Darknet coordinate scaling (2.0 - ground_truth_w * ground_truth_h)
        coord_scale = 2.0 - (twh.exp()[..., 0:1] / w) * (twh.exp()[..., 1:2] / h)
        
        # Loss components (.sum() produces a 0D scalar)
        loss_xy = (mask * coord_scale * (xy - txy).square()).sum()
        loss_wh = (mask * coord_scale * (wh - twh).square()).sum()
        loss_obj = (mask * (conf - tconf).square()).sum()
        loss_noobj = (noobj_mask * conf.square()).sum()
        loss_cls = (mask * (cls_prob - tcls).square()).sum()
        
        total_loss = total_loss + loss_xy + loss_wh + loss_obj + loss_noobj + loss_cls
        
    # Return 0D scalar normalized by batch size
    return total_loss / float(preds[0].shape[0])

# ==============================================================================
# 3. JIT-Fused Training Step
# ==============================================================================

# TinyJit compiles the complete backward, loss graph, and momentum update into GPU kernels
@TinyJit
def train_step_jit(model: YOLOv3, opt: nn.optim.SGD, x: Tensor, t0: Tensor, t1: Tensor, t2: Tensor) -> Tensor:
    Tensor.training = True
    preds = model(x)
    loss = yolov3_loss(preds, [t0, t1, t2])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.realize()

# ==============================================================================
# 4. pjreddie Darknet Learning Rate Scheduler
# ==============================================================================

def get_darknet_learning_rate(iteration: int, base_lr: float = 0.001, burn_in: int = 1000, 
                               steps: List[int] = [400000, 450000], scales: List[float] = [0.1, 0.1]) -> float:
    """Exact Darknet burn-in & step policy."""
    if iteration < burn_in:
        return base_lr * math.pow(iteration / burn_in, 4)
    lr = base_lr
    for step, scale in zip(steps, scales):
        if iteration >= step:
            lr *= scale
    return lr

# ==============================================================================
# 5. Synthetic Darknet Data Generator (for verification & benchmarking)
# ==============================================================================

def generate_mock_batch(batch_size: int, num_classes: int = 80, img_size: int = 416):
    images = np.random.uniform(0.0, 1.0, size=(batch_size, 3, img_size, img_size)).astype(np.float32)
    targets = []
    for scale_div in [32, 16, 8]:
        grid_dim = img_size // scale_div
        t = np.zeros((batch_size, 3, grid_dim, grid_dim, 7 + num_classes), dtype=np.float32)
        # Randomly assign 1 object per item in batch
        for b in range(batch_size):
            gx, gy = np.random.randint(0, grid_dim, size=2)
            anchor_idx = np.random.randint(0, 3)
            cls_idx = np.random.randint(0, num_classes)
            t[b, anchor_idx, gy, gx, 0] = 1.0        # mask
            t[b, anchor_idx, gy, gx, 1] = 0.0        # noobj_mask (0 for object)
            t[b, anchor_idx, gy, gx, 2:4] = [0.5, 0.5] # txy
            t[b, anchor_idx, gy, gx, 4:6] = [0.0, 0.0] # twh (log-space)
            t[b, anchor_idx, gy, gx, 6] = 1.0        # tconf
            t[b, anchor_idx, gy, gx, 7 + cls_idx] = 1.0
            # Initialize remaining as noobj
            t[b, :, :, :, 1] = 1.0 - t[b, :, :, :, 0]
        targets.append(t)
    return images, targets

# ==============================================================================
# 6. Main Darknet Training Loop
# ==============================================================================

def train():
    num_classes = 80
    base_lr = 0.001
    burn_in = 1000
    batch_size = 64
    subdivisions = 8
    sub_batch_size = batch_size // subdivisions
    max_batches = 500200
    
    print(f"[Darknet] Initializing YOLOv3 (Classes: {num_classes}) on {Device.DEFAULT}")
    model = YOLOv3(num_classes=num_classes)
    
    # pjreddie darknet default: SGD momentum=0.9, decay=0.0005
    params = nn.state.get_parameters(model)
    optimizer = nn.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0005)
    
    print(f"[Darknet] Training Config | Batch: {batch_size} | Subdivisions: {subdivisions} (Step size: {sub_batch_size})")
    print(f"[Darknet] JIT: Enabled | Burn-in: {burn_in} steps")
    
    global_iteration = 0
    t_start = time.perf_counter()
    
    while global_iteration < max_batches:
        global_iteration += 1
        
        # 1. Update Learning Rate (Darknet burn-in / step decay)
        current_lr = get_darknet_learning_rate(global_iteration, base_lr=base_lr, burn_in=burn_in)
        optimizer.lr.assign(current_lr)
        
        batch_loss = 0.0
        
        # 2. Process Batch across Subdivisions
        for sub in range(subdivisions):
            imgs_np, targets_np = generate_mock_batch(sub_batch_size, num_classes=num_classes, img_size=416)
            
            x = Tensor(imgs_np)
            t0 = Tensor(targets_np[0])
            t1 = Tensor(targets_np[1])
            t2 = Tensor(targets_np[2])
            
            sub_loss = train_step_jit(model, optimizer, x, t0, t1, t2)
            batch_loss += sub_loss.item()
        
        batch_loss /= subdivisions
        
        # Logging matching darknet format
        if global_iteration % 10 == 0 or global_iteration <= 20:
            elapsed = time.perf_counter() - t_start
            imgs_per_sec = (10 * batch_size) / elapsed if global_iteration > 1 else batch_size / elapsed
            print(f"{global_iteration:d}: {batch_loss:.4f}, {batch_loss:.4f} avg, {optimizer.lr.item():.6f} rate, "
                  f"{elapsed:.2f} seconds, {global_iteration * batch_size} images, {imgs_per_sec:.1f} img/s")
            t_start = time.perf_counter()

if __name__ == "__main__":
    train()
