"""Probe tinygrad devices and benchmark simple ops + small/medium models.

Usage:
  python tinygrad_bench.py
  python tinygrad_bench.py --device METAL
  python tinygrad_bench.py --ops-only
  python tinygrad_bench.py --models-only
  python tinygrad_bench.py --all-devices
  python tinygrad_bench.py --small
  python tinygrad_bench.py --json
  python tinygrad_bench.py --json results.json

"""

from __future__ import annotations

import argparse
import json
import pkgutil
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from tinygrad import Device, Tensor, TinyJit, nn

try:
    from tinygrad import Context, DEV
except ImportError:
    from tinygrad.helpers import Context, DEV  # type: ignore

from tinygrad.nn.state import get_parameters

# User priority: 1 CUDA|METAL|HIP  2 CL  3 CPU  4 WebGPU
# tinygrad names: NV = newer NVIDIA runtime, GPU = OpenCL (ops_gpu, removed in v0.13+)
PRIORITY_GROUPS: list[tuple[int, str, list[tuple[str, str]]]] = [
    (1, "CUDA/METAL/HIP", [
        ("CUDA", "CUDA"),
        ("NV", "CUDA"),
        ("METAL", "METAL"),
        ("HIP", "HIP"),
        ("AMD", "HIP"),
    ]),
    (2, "CL", [
        ("GPU", "CL"),
        ("CL", "CL"),
    ]),
    (3, "CPU", [
        ("CPU", "CPU"),
        ("LLVM", "CPU"),
    ]),
    (4, "WEBGPU", [
        ("WEBGPU", "WEBGPU"),
    ]),
]

KNOWN_ORDER = [name for _, _, aliases in PRIORITY_GROUPS for name, _ in aliases]

# Map runtime modules to Device names. ops_gpu is listed only so we can report its absence.
OPS_TO_DEVICE = {
    "ops_cuda": "CUDA",
    "ops_nv": "NV",
    "ops_metal": "METAL",
    "ops_hip": "HIP",
    "ops_amd": "AMD",
    "ops_gpu": "GPU",       # OpenCL — removed in v0.13+
    "ops_cpu": "CPU",
    "ops_webgpu": "WEBGPU",
    "ops_llvm": "LLVM",
    "ops_dsp": "DSP",
    "ops_qcom": "QCOM",
}

TARGET_MAJOR_MINOR = (0, 13)


@dataclass
class Knobs:
    warmup: int = 1
    runs: int = 3
    matmul_n: int = 2048
    add_elems: int = 16 * 1024 * 1024
    conv_n: int = 8
    conv_cin: int = 64
    conv_cout: int = 128
    conv_h: int = 56
    conv_w: int = 56
    conv_k: int = 3
    mnist_bs: int = 128
    alexnet_bs: int = 8
    alexnet_size: int = 224
    googlenet_bs: int = 4
    googlenet_size: int = 224

    @staticmethod
    def small() -> "Knobs":
        """Low-RAM preset: still the same graphs, just smaller tensors / batches."""
        return Knobs(
            matmul_n=1024,
            add_elems=4 * 1024 * 1024,
            conv_n=2,
            conv_cin=32,
            conv_cout=64,
            conv_h=28,
            conv_w=28,
            conv_k=3,
            mnist_bs=32,
            alexnet_bs=1,
            alexnet_size=224,
            googlenet_bs=1,
            googlenet_size=224,
        )


def tinygrad_version() -> str:
    try:
        from importlib.metadata import version
        return version("tinygrad")
    except Exception:
        return str(getattr(__import__("tinygrad"), "__version__", "unknown"))


def parse_version(v: str) -> tuple[int, int, int] | None:
    nums: list[int] = []
    for part in v.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            nums.append(int(digits))
        if len(nums) == 3:
            break
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def require_v013(ver: str, allow_newer: bool) -> None:
    parsed = parse_version(ver)
    print(f"target: tinygrad v0.13.x  (OpenCL ops_gpu removed; skip 0.14+ perf drop)")
    if parsed is None:
        print(f"warning: could not parse tinygrad version {ver!r}; continuing")
        return
    major, minor, patch = parsed
    if (major, minor) == TARGET_MAJOR_MINOR:
        return
    if (major, minor) >= (0, 14):
        msg = (
            f"this script targets tinygrad 0.13.x; you have {ver}. "
            "v0.14+ showed a large performance drop in our tests."
        )
        if allow_newer:
            print(f"warning: {msg} continuing because --allow-newer")
            return
        raise SystemExit(f"error: {msg}  re-run with --allow-newer if you really want this.")
    print(f"warning: tinygrad {ver} is older than 0.13.0; APIs may differ")


def current_default_device() -> str:
    """Read-only. Do not assign Device.DEFAULT — it is deprecated."""
    for attr in ("DEFAULT", "default"):
        if hasattr(Device, attr):
            try:
                return str(getattr(Device, attr))
            except Exception:
                pass
    return str(DEV.value)


def reported_devices() -> list[str]:
    if hasattr(Device, "get_available_devices"):
        try:
            return [str(x) for x in Device.get_available_devices()]
        except Exception:
            pass
    return []


def list_runtime_ops() -> list[str]:
    """Filenames under tinygrad.runtime (ops_*.py). Do not import them — ops_gpu is gone in v0.13."""
    try:
        import tinygrad.runtime as rt
        paths = getattr(rt, "__path__", None)
        if not paths:
            return []
        return sorted(m.name for m in pkgutil.iter_modules(paths) if m.name.startswith("ops_"))
    except Exception:
        return []


def sync(dev: str) -> None:
    try:
        Device[dev].synchronize()
    except Exception:
        pass


def probe(dev: str) -> tuple[bool, str]:
    """Return (ok, error_message). A listed device is not always usable (e.g. DSP)."""
    try:
        with Context(DEV=dev):
            t = Tensor.ones(16, device=dev).contiguous().realize()
            (t + 1).realize()
            sync(dev)
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")
        return False, msg[:160]


def bench(dev: str, fn: Callable[[], None], knobs: Knobs) -> tuple[float, float, float]:
    """Returns (warmup_s, avg_s, best_s)."""
    t0 = time.perf_counter()
    for _ in range(knobs.warmup):
        fn()
        sync(dev)
    warmup_s = (time.perf_counter() - t0) / max(knobs.warmup, 1)

    samples: list[float] = []
    for _ in range(knobs.runs):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        samples.append(time.perf_counter() - t0)
    return warmup_s, statistics.mean(samples), min(samples)


def fmt_ms(s: float) -> str:
    return f"{s * 1000:10.2f} ms"


def fmt_thr(value: float, unit: str) -> str:
    return f"{value:10.2f} {unit}"


def result(
    *,
    device: str,
    family: str,
    kind: str,
    name: str,
    shape: str,
    warmup: float,
    avg: float,
    best: float,
    thr_avg: float,
    thr_best: float,
    unit: str,
    extra: str,
    ok: bool = True,
    error: str = "",
) -> dict[str, Any]:
    return dict(
        device=device, family=family, kind=kind, name=name, shape=shape,
        warmup_s=warmup, avg_s=avg, best_s=best,
        thr_avg=thr_avg, thr_best=thr_best, unit=unit,
        extra=extra, ok=ok, error=error,
    )


def fail_result(device: str, family: str, kind: str, name: str, err: Exception) -> dict[str, Any]:
    return result(
        device=device, family=family, kind=kind, name=name, shape="",
        warmup=0, avg=0, best=0, thr_avg=0, thr_best=0, unit="",
        extra="", ok=False, error=f"{type(err).__name__}: {err}",
    )


# --- simple ops ---

def run_add(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    n = knobs.add_elems
    a = Tensor.randn(n, device=dev).realize()
    b = Tensor.randn(n, device=dev).realize()
    warmup, avg, best = bench(dev, lambda: (a + b).realize(), knobs)
    gb = 3 * n * 4 / 1e9  # 2 reads + 1 write, float32
    return result(
        device=dev, family=family, kind="op", name="add", shape=f"{n}",
        warmup=warmup, avg=avg, best=best,
        thr_avg=gb / avg, thr_best=gb / best, unit="GB/s",
        extra=f"{n/1e6:.0f}M f32  (memory-bound)",
    )


def run_matmul(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    n = knobs.matmul_n
    a = Tensor.randn(n, n, device=dev).realize()
    b = Tensor.randn(n, n, device=dev).realize()
    warmup, avg, best = bench(dev, lambda: (a @ b).realize(), knobs)
    flops = 2.0 * n * n * n
    return result(
        device=dev, family=family, kind="op", name="matmul",
        shape=f"{n}x{n}@{n}x{n}",
        warmup=warmup, avg=avg, best=best,
        thr_avg=flops / avg / 1e9, thr_best=flops / best / 1e9, unit="GFLOPS",
        extra=f"{flops/1e9:.2f} GFLOP / run",
    )


def run_conv2d(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    n, cin, cout, h, w, k = (
        knobs.conv_n, knobs.conv_cin, knobs.conv_cout,
        knobs.conv_h, knobs.conv_w, knobs.conv_k,
    )
    x = Tensor.randn(n, cin, h, w, device=dev).realize()
    weight = Tensor.randn(cout, cin, k, k, device=dev).realize()
    pad = k // 2
    warmup, avg, best = bench(dev, lambda: x.conv2d(weight, padding=pad).realize(), knobs)
    flops = 2.0 * n * cout * h * w * cin * k * k
    return result(
        device=dev, family=family, kind="op", name="conv2d",
        shape=f"x{n,cin,h,w} w{cout,cin,k,k}",
        warmup=warmup, avg=avg, best=best,
        thr_avg=flops / avg / 1e9, thr_best=flops / best / 1e9, unit="GFLOPS",
        extra=f"{flops/1e9:.2f} GFLOP / run  pad={pad}",
    )


# --- models (self-contained; no tinygrad.extra / no ops_gpu) ---

class MnistCNN:
    def __init__(self):
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c1(x).relu().max_pool2d(2)
        x = self.c2(x).relu().max_pool2d(2)
        return self.fc(x.flatten(1))


class AlexNet:
    """Classic AlexNet (224 input). Dropout stripped so inference is deterministic."""

    def __init__(self, num_classes: int = 1000):
        self.c1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.c2 = nn.Conv2d(64, 192, 5, padding=2)
        self.c3 = nn.Conv2d(192, 384, 3, padding=1)
        self.c4 = nn.Conv2d(384, 256, 3, padding=1)
        self.c5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c1(x).relu().max_pool2d(3, stride=2)
        x = self.c2(x).relu().max_pool2d(3, stride=2)
        x = self.c3(x).relu()
        x = self.c4(x).relu()
        x = self.c5(x).relu().max_pool2d(3, stride=2)
        x = x.flatten(1)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.fc3(x)


class Inception:
    """GoogLeNet Inception module (no aux classifier)."""

    def __init__(self, cin: int, ch1x1: int, ch3x3red: int, ch3x3: int,
                 ch5x5red: int, ch5x5: int, pool_proj: int):
        self.b1 = nn.Conv2d(cin, ch1x1, 1)
        self.b2_1 = nn.Conv2d(cin, ch3x3red, 1)
        self.b2_2 = nn.Conv2d(ch3x3red, ch3x3, 3, padding=1)
        self.b3_1 = nn.Conv2d(cin, ch5x5red, 1)
        self.b3_2 = nn.Conv2d(ch5x5red, ch5x5, 5, padding=2)
        self.b4 = nn.Conv2d(cin, pool_proj, 1)

    def __call__(self, x: Tensor) -> Tensor:
        b1 = self.b1(x).relu()
        b2 = self.b2_2(self.b2_1(x).relu()).relu()
        b3 = self.b3_2(self.b3_1(x).relu()).relu()
        b4 = self.b4(x.max_pool2d(3, stride=1, padding=1)).relu()
        return Tensor.cat(b1, b2, b3, b4, dim=1)


class GoogLeNet:
    """Inception v1 / GoogLeNet, inference only (aux heads omitted). Input 3x224x224."""

    def __init__(self, num_classes: int = 1000):
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 192, 3, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.fc = nn.Linear(1024, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu().max_pool2d(3, stride=2, padding=1)
        x = self.conv2(x).relu()
        x = self.conv3(x).relu().max_pool2d(3, stride=2, padding=1)
        x = self.inception3a(x)
        x = self.inception3b(x).max_pool2d(3, stride=2, padding=1)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x).max_pool2d(3, stride=2, padding=1)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = x.avg_pool2d(7).flatten(1)
        return self.fc(x)


def realize_params(model) -> None:
    for p in get_parameters(model):
        p.realize()


def conv_flops(n: int, cin: int, cout: int, hout: int, wout: int, k: int) -> float:
    return 2.0 * n * cout * hout * wout * cin * k * k


def inception_flops(n: int, cin: int, h: int, w: int,
                    ch1x1: int, ch3x3red: int, ch3x3: int,
                    ch5x5red: int, ch5x5: int, pool_proj: int) -> float:
    return (
        conv_flops(n, cin, ch1x1, h, w, 1)
        + conv_flops(n, cin, ch3x3red, h, w, 1)
        + conv_flops(n, ch3x3red, ch3x3, h, w, 3)
        + conv_flops(n, cin, ch5x5red, h, w, 1)
        + conv_flops(n, ch5x5red, ch5x5, h, w, 5)
        + conv_flops(n, cin, pool_proj, h, w, 1)
    )


def mnist_flops(bs: int) -> float:
    conv1 = 2.0 * bs * 32 * 26 * 26 * 1 * 3 * 3
    conv2 = 2.0 * bs * 64 * 11 * 11 * 32 * 3 * 3
    fc = 2.0 * bs * 1600 * 10
    return conv1 + conv2 + fc


def alexnet_flops(bs: int) -> float:
    f = 0.0
    f += conv_flops(bs, 3, 64, 55, 55, 11)
    f += conv_flops(bs, 64, 192, 27, 27, 5)
    f += conv_flops(bs, 192, 384, 13, 13, 3)
    f += conv_flops(bs, 384, 256, 13, 13, 3)
    f += conv_flops(bs, 256, 256, 13, 13, 3)
    f += 2.0 * bs * (256 * 6 * 6) * 4096
    f += 2.0 * bs * 4096 * 4096
    f += 2.0 * bs * 4096 * 1000
    return f


def googlenet_flops(bs: int) -> float:
    f = 0.0
    f += conv_flops(bs, 3, 64, 112, 112, 7)
    f += conv_flops(bs, 64, 64, 56, 56, 1)
    f += conv_flops(bs, 64, 192, 56, 56, 3)
    f += inception_flops(bs, 192, 28, 28, 64, 96, 128, 16, 32, 32)
    f += inception_flops(bs, 256, 28, 28, 128, 128, 192, 32, 96, 64)
    f += inception_flops(bs, 480, 14, 14, 192, 96, 208, 16, 48, 64)
    f += inception_flops(bs, 512, 14, 14, 160, 112, 224, 24, 64, 64)
    f += inception_flops(bs, 512, 14, 14, 128, 128, 256, 24, 64, 64)
    f += inception_flops(bs, 512, 14, 14, 112, 144, 288, 32, 64, 64)
    f += inception_flops(bs, 528, 14, 14, 256, 160, 320, 32, 128, 128)
    f += inception_flops(bs, 832, 7, 7, 256, 160, 320, 32, 128, 128)
    f += inception_flops(bs, 832, 7, 7, 384, 192, 384, 48, 128, 128)
    f += 2.0 * bs * 1024 * 1000
    return f


def run_model(dev: str, family: str, name: str, model, x: Tensor,
              flops: float, knobs: Knobs, extra_note: str = "") -> dict[str, Any]:
    realize_params(model)
    x = x.realize()
    warmup, avg, best = bench(dev, lambda: model(x).realize(), knobs)
    bs = int(x.shape[0])
    note = f"~{flops/1e9:.2f} GFLOP/batch  |  {flops/avg/1e9:.1f} GFLOPS avg"
    if extra_note:
        note += f"  |  {extra_note}"
    return result(
        device=dev, family=family, kind="model", name=name,
        shape=str(tuple(int(s) for s in x.shape)),
        warmup=warmup, avg=avg, best=best,
        thr_avg=bs / avg, thr_best=bs / best, unit="samp/s",
        extra=note,
    )


def run_mnist(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    model = MnistCNN()
    x = Tensor.randn(knobs.mnist_bs, 1, 28, 28, device=dev)
    return run_model(dev, family, "mnist_cnn", model, x, mnist_flops(knobs.mnist_bs), knobs)


def run_mnist_jit(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    model = MnistCNN()
    realize_params(model)
    x = Tensor.randn(knobs.mnist_bs, 1, 28, 28, device=dev).realize()

    @TinyJit
    def fwd():
        return model(x).realize()

    warmup, avg, best = bench(dev, fwd, knobs)
    flops = mnist_flops(knobs.mnist_bs)
    return result(
        device=dev, family=family, kind="model", name="mnist_cnn+jit",
        shape=str(tuple(int(s) for s in x.shape)),
        warmup=warmup, avg=avg, best=best,
        thr_avg=knobs.mnist_bs / avg, thr_best=knobs.mnist_bs / best, unit="samp/s",
        extra=f"~{flops/1e9:.2f} GFLOP/batch  |  TinyJit replay",
    )


def run_alexnet(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    model = AlexNet()
    x = Tensor.randn(knobs.alexnet_bs, 3, knobs.alexnet_size, knobs.alexnet_size, device=dev)
    return run_model(dev, family, "alexnet", model, x, alexnet_flops(knobs.alexnet_bs), knobs)


def run_googlenet(dev: str, family: str, knobs: Knobs) -> dict[str, Any]:
    model = GoogLeNet()
    x = Tensor.randn(knobs.googlenet_bs, 3, knobs.googlenet_size, knobs.googlenet_size, device=dev)
    return run_model(
        dev, family, "googlenet", model, x, googlenet_flops(knobs.googlenet_bs), knobs,
        extra_note="Inception v1, no aux",
    )


# --- reporting ---

def print_row(r: dict[str, Any]) -> None:
    if not r["ok"]:
        print(f"  {r['name']:<14} FAILED  {r['error']}")
        return
    print(
        f"  {r['name']:<14} {r['shape']:<28} "
        f"warmup {fmt_ms(r['warmup_s'])}  "
        f"avg {fmt_ms(r['avg_s'])}  "
        f"best {fmt_ms(r['best_s'])}  "
        f"avg {fmt_thr(r['thr_avg'], r['unit'])}  "
        f"best {fmt_thr(r['thr_best'], r['unit'])}"
    )
    print(f"  {'':<14} {r['extra']}")


def run_one(fn: Callable, dev: str, family: str, knobs: Knobs,
            kind: str, name: str) -> dict[str, Any]:
    try:
        return fn(dev, family, knobs)
    except Exception as e:
        r = fail_result(dev, family, kind, name, e)
        print_row(r)
        return r


def run_suite(dev: str, family: str, knobs: Knobs, ops: bool, models: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    print(f"\n=== {dev}  (DEV={DEV.value}) ===")
    with Context(DEV=dev):
        print(f"  inside Context: DEV={DEV.value}  Device.DEFAULT={current_default_device()}")
        if ops:
            print("\n  simple ops")
            for fn, name in (
                (run_add, "add"),
                (run_matmul, "matmul"),
                (run_conv2d, "conv2d"),
            ):
                r = run_one(fn, dev, family, knobs, "op", name)
                if r["ok"]:
                    print_row(r)
                rows.append(r)
        if models:
            print("\n  models  (inference, no TinyJit unless noted)")
            for fn, name in (
                (run_mnist, "mnist_cnn"),
                (run_mnist_jit, "mnist_cnn+jit"),
                (run_alexnet, "alexnet"),
                (run_googlenet, "googlenet"),
            ):
                r = run_one(fn, dev, family, knobs, "model", name)
                if r["ok"]:
                    print_row(r)
                rows.append(r)
    return rows


def select_devices(working: dict[str, str], want: str | None) -> tuple[str, list[str]]:
    if want:
        key = want.upper()
        aliases = {name.upper(): name for name in working}
        if key in aliases:
            return aliases[key], [aliases[key]]
        family_hits = [d for d, fam in working.items() if fam.upper() == key]
        if family_hits:
            return family_hits[0], family_hits
        raise SystemExit(f"device {want!r} is not available. working: {list(working)}")

    for _, _, aliases in PRIORITY_GROUPS:
        for name, _fam in aliases:
            if name in working:
                return name, list(working)
    raise SystemExit("no working devices in the priority list")


def write_json(path: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2)
    if path == "-":
        print("\n----- json -----")
        print(text)
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    print(f"\njson written to {path}")


def build_knobs(args: argparse.Namespace) -> Knobs:
    knobs = Knobs.small() if args.small else Knobs()
    if args.warmup is not None:
        knobs = replace(knobs, warmup=args.warmup)
    if args.runs is not None:
        knobs = replace(knobs, runs=args.runs)
    if args.matmul_n is not None:
        knobs = replace(knobs, matmul_n=args.matmul_n)
    if args.mnist_bs is not None:
        knobs = replace(knobs, mnist_bs=args.mnist_bs)
    if args.alexnet_bs is not None:
        knobs = replace(knobs, alexnet_bs=args.alexnet_bs)
    if args.googlenet_bs is not None:
        knobs = replace(knobs, googlenet_bs=args.googlenet_bs)
    return knobs


def main() -> None:
    p = argparse.ArgumentParser(description="tinygrad v0.13 device probe + benchmark")
    p.add_argument("--device", help="run only this device / family (e.g. CUDA, METAL, CPU, CL)")
    p.add_argument("--ops-only", action="store_true")
    p.add_argument("--models-only", action="store_true")
    p.add_argument("--all-devices", action="store_true",
                   help="run models on every working device, not just primary")
    p.add_argument("--small", action="store_true",
                   help="low-RAM preset: 1024 matmul, smaller conv, batch 1 for AlexNet/GoogLeNet")
    p.add_argument("--warmup", type=int, default=None, help="warmup runs (default 1)")
    p.add_argument("--runs", type=int, default=None, help="timed runs (default 3)")
    p.add_argument("--matmul-n", type=int, default=None, dest="matmul_n")
    p.add_argument("--mnist-bs", type=int, default=None, dest="mnist_bs")
    p.add_argument("--alexnet-bs", type=int, default=None, dest="alexnet_bs")
    p.add_argument("--googlenet-bs", type=int, default=None, dest="googlenet_bs")
    p.add_argument("--json", nargs="?", const="tinygrad_bench.json", metavar="PATH",
                   help="write results JSON to PATH (default: tinygrad_bench.json; '-' = stdout)")
    p.add_argument("--allow-newer", action="store_true",
                   help="run even if tinygrad is 0.14+ (known perf drop)")
    args = p.parse_args()
    ops = not args.models_only
    models = not args.ops_only
    knobs = build_knobs(args)

    ver = tinygrad_version()
    print(f"tinygrad {ver}  |  {platform.system()} {platform.machine()}  |  python {platform.python_version()}")
    require_v013(ver, args.allow_newer)
    print(f"DEV.value = {DEV.value}  |  Device.DEFAULT = {current_default_device()}")
    print(f"timing: warmup={knobs.warmup}  timed runs={knobs.runs}  (avg + best)"
          f"{'  [small preset]' if args.small else ''}")
    print(
        f"sizes: matmul {knobs.matmul_n}  add {knobs.add_elems/1e6:.0f}M  "
        f"conv n={knobs.conv_n}  mnist_bs={knobs.mnist_bs}  "
        f"alexnet_bs={knobs.alexnet_bs}  googlenet_bs={knobs.googlenet_bs}"
    )

    runtime_ops = list_runtime_ops()
    print("\nruntime modules:", ", ".join(runtime_ops) if runtime_ops else "(none found)")
    if "ops_gpu" in runtime_ops:
        print("note: ops_gpu is present on this install (OpenCL device GPU)")
    else:
        print("note: ops_gpu (OpenCL, device GPU) is absent — removed in tinygrad v0.13+")
        print("      CL probe below is expected to FAIL; that is not a broken install.")

    reported = reported_devices()
    print("\nreported by tinygrad:", ", ".join(reported) if reported else "(get_available_devices() not available)")

    extra = [d for d in reported if d not in KNOWN_ORDER]
    to_probe = list(dict.fromkeys(KNOWN_ORDER + extra))

    print("\nprobe")
    working: dict[str, str] = {}
    family_of = {name: fam for _, _, aliases in PRIORITY_GROUPS for name, fam in aliases}
    probe_log: list[dict[str, Any]] = []
    for name in to_probe:
        ok, err = probe(name)
        fam = family_of.get(name, "other")
        probe_log.append({"device": name, "family": fam, "ok": ok, "error": err})
        if ok:
            working[name] = fam
            print(f"  {name:<10} {fam:<14} \033[32m OK \033[0m ")
        else:
            print(f"  {name:<10} {fam:<14} \033[31m FAIL \033[0m {err}")

    if not working:
        raise SystemExit("no device could run a 16-element tensor")

    primary, bench_devs = select_devices(working, args.device)
    print(f"\nselected primary: {primary}  (family {working[primary]})")
    print("benchmark devices:", ", ".join(bench_devs))
    print("WebGPU note: needs a working wgpu/dawn stack; FAIL above is normal if it is not installed.")

    all_rows: list[dict[str, Any]] = []
    for dev in bench_devs:
        run_models = models and (args.all_devices or args.device or dev == primary)
        all_rows.extend(run_suite(dev, working[dev], knobs, ops=ops, models=run_models))

    payload = {
        "tinygrad": ver,
        "target": "0.13.x",
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "dev_default": current_default_device(),
        "runtime_ops": runtime_ops,
        "ops_gpu_present": "ops_gpu" in runtime_ops,
        "knobs": asdict(knobs),
        "primary": primary,
        "working_devices": working,
        "probe": probe_log,
        "results": all_rows,
    }
    if args.json:
        write_json(args.json, payload)

    print("\ndone.")


if __name__ == "__main__":
    main()
