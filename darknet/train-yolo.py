import argparse
import os
import struct
from PIL import Image
import numpy as np
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn import optim
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

NUM_CLASSES = 20
INPUT_SIZE = 416
MULTI_SCALE_SIZES = [320, 416, 512, 608]
ANCHORS = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]], dtype=np.float32)
CHECKPOINT_PATH = "yolov3.safetensors"

def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-16)

def giou_loss(pred_boxes, target_boxes):
    x1, y1, w1, h1 = pred_boxes.T
    x2, y2, w2, h2 = target_boxes.T
    xi1 = np.maximum(x1, x2)
    yi1 = np.maximum(y1, y2)
    xi2 = np.minimum(x1 + w1, x2 + w2)
    yi2 = np.minimum(y1 + h1, y2 + h2)
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-16)
    xc1 = np.minimum(x1, x2)
    yc1 = np.minimum(y1, y2)
    xc2 = np.maximum(x1 + w1, x2 + w2)
    yc2 = np.maximum(y1 + h1, y2 + h2)
    c_area = (xc2 - xc1) * (yc2 - yc1)
    giou = iou - (c_area - union) / (c_area + 1e-16)
    return (1 - giou).mean()

def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        ious = np.array([bbox_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep

def compute_map(preds, targets, num_classes=NUM_CLASSES, iou_threshold=0.5):
    """Simplified mAP calculation"""
    aps = []
    for cls in range(num_classes):
        pred_boxes, pred_scores, gt_boxes = [], [], []
        for p, t in zip(preds, targets):
            for scale in p:
                for y in range(scale.shape[0]):
                    for x in range(scale.shape[1]):
                        for a in range(3):
                            if scale[y, x, a, 4] > 0.5:
                                pred_boxes.append(scale[y, x, a, :4])
                                pred_scores.append(scale[y, x, a, 4])
            for box in t:
                if int(box[0]) == cls:
                    gt_boxes.append(box[1:5])
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        keep = nms(np.array(pred_boxes), np.array(pred_scores))
        tp, fp = 0, 0
        for idx in keep:
            ious = [bbox_iou(pred_boxes[idx], g) for g in gt_boxes]
            if max(ious) >= iou_threshold:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp + 1e-6)
        recall = tp / len(gt_boxes)
        aps.append(precision * recall)
    return np.mean(aps) if aps else 0.0

class ResidualBlock():
    def __init__(self, channels):
        self.conv1 = nn.Conv2d(channels, channels//2, 1)
        self.conv2 = nn.Conv2d(channels//2, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels//2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x).relu()
        x = self.bn1(x)
        x = self.conv2(x).relu()
        x = self.bn2(x)
        return x + residual

class Darknet53():
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.res1 = Tensor.sequential(*[ResidualBlock(64) for _ in range(1)])
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res2 = Tensor.sequential(*[ResidualBlock(128) for _ in range(2)])
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.res3 = Tensor.sequential(*[ResidualBlock(256) for _ in range(8)])
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.res4 = Tensor.sequential(*[ResidualBlock(512) for _ in range(8)])
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.res5 = Tensor.sequential(*[ResidualBlock(1024) for _ in range(4)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.res1(x)
        x = self.conv3(x).relu()
        x = self.res2(x)
        x = self.conv4(x).relu()
        x = self.res3(x)
        route1 = x
        x = self.conv5(x).relu()
        x = self.res4(x)
        route2 = x
        x = self.conv6(x).relu()
        x = self.res5(x)
        return x, route1, route2

class YOLOv3:
    def __init__(self):
        self.backbone = Darknet53()
        self.yolo1 = nn.Conv2d(1024, (NUM_CLASSES + 5) * 3, 1)
        self.yolo2 = nn.Conv2d(512, (NUM_CLASSES + 5) * 3, 1)
        self.yolo3 = nn.Conv2d(256, (NUM_CLASSES + 5) * 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        x, route1, route2 = self.backbone(x)
        return [self.yolo1(x), self.yolo2(route2), self.yolo3(route1)]

def load_darknet_weights(model, weights_path):
    with open(weights_path, "rb") as f:
        major, minor, revision, seen = struct.unpack("<iiii", f.read(16))
        print(f"Darknet version: {major}.{minor}.{revision} (seen={seen})")
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                if layer.bias is not None:
                    layer.bias = Tensor(np.fromfile(f, dtype=np.float32, count=layer.out_channels))
                layer.weight = Tensor(np.fromfile(f, dtype=np.float32, count=layer.weight.numel()).reshape(layer.weight.shape))
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight = Tensor(np.fromfile(f, dtype=np.float32, count=layer.num_features))
                layer.bias = Tensor(np.fromfile(f, dtype=np.float32, count=layer.num_features))
                layer.running_mean = Tensor(np.fromfile(f, dtype=np.float32, count=layer.num_features))
                layer.running_var = Tensor(np.fromfile(f, dtype=np.float32, count=layer.num_features))

def encode_targets(boxes, input_size, anchors, num_classes):
    targets = [np.zeros((s, s, 3, 5 + num_classes), dtype=np.float32) for s in [13, 26, 52]]
    for box in boxes:
        cls, cx, cy, w, h = box
        best_iou, best_anchor = 0, 0
        for i, anchor in enumerate(anchors):
            iou = bbox_iou([0, 0, w, h], [0, 0, anchor[0]/input_size, anchor[1]/input_size])
            if iou > best_iou:
                best_iou = iou
                best_anchor = i
        scale_idx = best_anchor // 3
        scale = [13, 26, 52][scale_idx]
        grid_x, grid_y = int(cx * scale), int(cy * scale)
        if 0 <= grid_x < scale and 0 <= grid_y < scale:
            targets[scale_idx][grid_y, grid_x, best_anchor % 3, :5] = [cx, cy, w, h, 1.0]
            targets[scale_idx][grid_y, grid_x, best_anchor % 3, 5 + int(cls)] = 1.0
    return targets

class YOLODataset:
    def __init__(self, image_paths, label_paths, multi_scale=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.multi_scale = multi_scale
        self.sizes = MULTI_SCALE_SIZES if multi_scale else [INPUT_SIZE]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"), dtype=np.float32) / 255.0
        boxes = []
        if os.path.exists(self.label_paths[idx]):
            with open(self.label_paths[idx]) as f:
                for line in f:
                    c, x, y, w, h = map(float, line.strip().split())
                    boxes.append([int(c), x, y, w, h])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5))
        size = np.random.choice(self.sizes) if self.multi_scale else INPUT_SIZE
        img = self._resize(img, size)
        boxes = self._resize_boxes(boxes, img.shape[:2], size)
        targets = encode_targets(boxes, size, ANCHORS, NUM_CLASSES)
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            if len(boxes): boxes[:, 1] = 1 - boxes[:, 1]
        img = self._pad_to_square(img, size)
        return img, targets, boxes

    def _resize(self, img, size):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        return np.array(Image.fromarray((img*255).astype(np.uint8)).resize((int(w*scale), int(h*scale))), dtype=np.float32) / 255.0

    def _resize_boxes(self, boxes, shape, size):
        if len(boxes) == 0: return boxes
        scale = size / max(shape)
        boxes[:, 1:] *= scale
        return boxes

    def _pad_to_square(self, img, size):
        h, w = img.shape[:2]
        pad_h, pad_w = (size - h) // 2, (size - w) // 2
        padded = np.zeros((size, size, 3), dtype=np.float32)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = img
        return padded

def create_dataloader(image_paths, label_paths, batch_size=4, train=True):
    dataset = YOLODataset(image_paths, label_paths)
    for i in range(0, len(dataset), batch_size):
        batch_imgs, batch_targets, batch_boxes = [], [], []
        for j in range(min(batch_size, len(dataset) - i)):
            img, targets, boxes = dataset[i + j]
            batch_imgs.append(img)
            batch_targets.append(targets)
            batch_boxes.append(boxes)
        yield Tensor(np.stack(batch_imgs)).permute(0, 3, 1, 2), batch_targets, batch_boxes

def yolo_loss(preds, targets):
    total = Tensor.zeros(1)
    for pred, target in zip(preds, targets):
        pred_np = pred.numpy()
        target_np = np.array(target)
        box_loss = giou_loss(pred_np[..., :4].reshape(-1, 4), target_np[..., :4].reshape(-1, 4))
        obj_loss = ((pred_np[..., 4:5] - target_np[..., 4:5]) ** 2).mean()
        cls_loss = ((pred_np[..., 5:] - target_np[..., 5:]) ** 2).mean()
        total = total + Tensor([box_loss + obj_loss + cls_loss])
    return total

@TinyJit
def train_step(model, optimizer, x, targets):
    optimizer.zero_grad()
    preds = model(x)
    loss = yolo_loss(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.realize()

def evaluate(model, image_paths, label_paths, batch_size=4):
    model.eval()
    all_preds, all_targets = [], []
    for x, targets, _ in create_dataloader(image_paths, label_paths, batch_size, train=False):
        preds = model(x)
        all_preds.append([p.numpy() for p in preds])
        all_targets.append(targets)
    map_score = compute_map(all_preds, all_targets)
    print(f"Validation mAP: {map_score:.4f}")
    model.train()
    return map_score

def train(args):
    model = YOLOv3()
    if args.weight and os.path.exists(args.weight):
        load_darknet_weights(model, args.weight)
    if args.checkpoint and os.path.exists(args.checkpoint):
        # model.load_state_dict(load_file(args.checkpoint))
        load_state_dict(model, safe_load(args.checkpoint))
        print(f"Resumed from {args.checkpoint}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    image_paths = ["data/VOCdevkit/VOC2012/JPEGImages/000001.jpg"]
    label_paths = ["data/VOCdevkit/VOC2012/labels/000001.txt"]

    for epoch in range(100):
        for batch_idx, (x, targets, _) in enumerate(create_dataloader(image_paths, label_paths)):
            loss = train_step(model, optimizer, x, targets)
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.numpy():.4f}")
        if epoch % 5 == 0:
            evaluate(model, image_paths, label_paths)
        safe_save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="voc.data")
    parser.add_argument("--weight", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    train(args)
