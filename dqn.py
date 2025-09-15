from tinygrad import Tensor
from tinygrad import Device


# set of actions + states
transtion = {}

def test():
  print("begintest")
  x = Tensor.eye(3, requires_grad=True)
  y = Tensor([[2.0,0,-2.0]], requires_grad=True)
  z = y.matmul(x).sum()
  z.backward()
  print(x.grad.tolist())  # dz/dx
  print(y.grad.tolist())  # dz/dy
  print("endtest")

# TODO: DQN implementation :: https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning
# how complex it should be to fit most training environment?
class DQN:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
    self.l3 = Tensor.kaiming_uniform(128, 10)
    self.l3 = Tensor.kaiming_uniform(128, 10)
    self.l3 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x: Tensor) -> Tensor:
    return x.flatten(1). dot(self.l1).relu().dot(self.l2)

class ExperienceReplay:
  def __init__(self):
    self.memory = deque([], maxlen=capacity)
  def push(self, *args):
    self.memory.append(transtion(*args))
  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
  def __len__(self):
    return len(self.memory)

if __name__ == "__main__":
  print("Running on " + Device.DEFAULT)
  test()
  model = DQN()
