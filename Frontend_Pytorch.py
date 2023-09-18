import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor

import numpy as np

from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

import torch
import torchvision
import torch.nn as nn


# Define a simple MLP in PyTorch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a model instance and a random input
model = MLP(input_size=784, hidden_size=256, output_size=10)
model.eval()
input_data = torch.randn(1, 784)

pytorch_output = model(input_data)
print("golden_pytorch: ", pytorch_output)

# Convert the PyTorch model to TorchScript format
scripted_model = torch.jit.trace(model, input_data).eval()

# Convert the PyTorch model to TVM
input_name = "input0"
shape_list = [(input_name, input_data.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Compile the TVM module
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# create module
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

# Set inputs
dtype = "float32"
input_data = np.asarray(input_data).astype(dtype)
module.set_input(input_name, input_data)
# Execute
module.run()
# Get outputs
tvm_output = module.get_output(0)
print("TVM_output: ",tvm_output)

