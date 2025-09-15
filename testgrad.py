from tinygrad import Tensor, nn, Device

# params
batchsize = 128
tau = 0.005
lr = 0.003

# what's agent currently doing?
state = {}

# what's agent can do?
action = {}

class DQN:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

class ExperienceReplay:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

if __name__ == "__main__":
  print(f"tinygrad is running on {Device.DEFAULT}")
  model = DQN()
  optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

  x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

  with Tensor.train():
    for i in range(10000):
      optim.zero_grad()
      loss = model(x).sparse_categorical_crossentropy(y).backward()
      optim.step()
      print(i, loss.item())

