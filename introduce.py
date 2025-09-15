from tinygrad import Tensor, Device

def test():
  a = Tensor.empty(4, 4)
  b = Tensor.empty(4, 4)
  print("Running on "+Device.DEFAULT)
  print(a+b)


if __name__ == "__main__":
  test()
