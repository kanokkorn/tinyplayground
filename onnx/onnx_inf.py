import sys, time
from tinygrad import TinyJit, GlobalCounters, fetch, getenv
from tinygrad.nn.onnx import OnnxRunner
from onnx_helpers import get_example_inputs, validate


def load_onnx_model(onnx_file):
  run_onnx = OnnxRunner(onnx_file)
  run_onnx_jit = TinyJit(lambda ** kwargs: next(iter(run_onnx({k:v.to(None) for k,v in kwargs.items()}).values())), prune=True)
  return run_onnx_jit, run_onnx.graph_inputs


if __name__ == "__main__":
  onnx_file = fetch(sys.argv[1])
  run_onnx_jit, input_specs = load_onnx_model(onnx_file)
  print("loaded model")
  inputs = get_example_inputs(input_specs)
  out = run_onnx_jit(**inputs)
  val = out.numpy()
  print(val)
