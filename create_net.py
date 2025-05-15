import torch
import numpy as np
class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.lin = torch.nn.Linear(2, 3)
        with torch.no_grad(): # fill with 1 for reproducibility
            self.lin.weight.fill_(1.0)
            self.lin.bias.fill_(1.0)
    def forward(self, x):
        x = self.lin(x)
        return x
    
net = TestNet()
net = net.double()
net.eval()
x = torch.tensor([3.0,5.0], dtype=torch.float64).reshape(1,2)
y = net(x).detach()
np.set_printoptions(precision=4, suppress=True, sign='+')
x, y = x.numpy().reshape(-1), y.numpy().reshape(-1)
print(f'x -> {x}')
print(f'y -> {y}')

#save the model
torch.jit.script(net).save('net.pt')

# convert to onnx
dummy_input = torch.randn(1, 2, dtype=torch.float64)
torch.onnx.export(net, dummy_input, 'net.onnx', export_params=True,
                  opset_version=12, do_constant_folding=True,
                  input_names=['x'], output_names=['y'])

