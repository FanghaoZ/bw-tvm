import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor

import torch.nn as nn
import torch.onnx

import numpy as np
import onnxruntime as rt

######################################################################
# Generated MLP ONNX model
# ---------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = MLP(input_size=784, hidden_size=256, output_size=10)

# Create a dummy input tensor
dummy_input = torch.randn(1, 784)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "mlp.onnx")

######################################################################
# Load ONNX model
onnx_model = onnx.load('mlp.onnx')
input_name = 'onnx::Gemm_0'

# Convert the ONNX model to Relay IR
input_shape = {input_name: (1, 784)}
mod, params = relay.frontend.from_onnx(onnx_model, shape=input_shape)

# Compile the model with relay
target = 'llvm'
    
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
    
# Execute on TVM
dtype = "float32"
input_data = np.asarray(dummy_input).astype("float32")
module.set_input(input_name, input_data)
module.run()
output_shape = (1, 10)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

print("TVM_output: ", tvm_output)

######################################################################
# Golden reference from ONNX runtime
sess = rt.InferenceSession("mlp.onnx", providers=rt.get_available_providers())

input_name0 = sess.get_inputs()[0].name

golden_onnx = sess.run(None, {input_name0: input_data})
print("golden_onnx: ", np.array(golden_onnx).flatten())

######################################################################
# Calculate the MSE
loss = np.square(np.subtract(tvm_output, golden_onnx)).mean()

print("MSE: ", loss)





