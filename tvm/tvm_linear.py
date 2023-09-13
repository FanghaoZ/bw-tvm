import sys
from pathlib import Path
BRAINSLICE_PATH = 'D:/BrainSlice-repo/develop/BrainSlice/'
EMULATOR_PATH = BRAINSLICE_PATH + 'target/distrib/retail/x64/app/BrainSlice/DevKit/lib/native/python'
SKU_PATH = BRAINSLICE_PATH + 'src/config/skugen/obj/amd64/BERT-NP/SKU.json'
FIRMWARE_LIB_PATH = BRAINSLICE_PATH + 'target/distrib/retail/x64/app/BrainSlice/Firmware/content'
FIRMWARE_PATH = FIRMWARE_LIB_PATH + '/BERT'
sys.path.append(EMULATOR_PATH)
sys.path.append(FIRMWARE_LIB_PATH)
import brainslice_client as bs_client
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
import torch
import numpy as np
from ISA import Mem
import Bert

def create_fpga_session(mode='fp32', sku_dir=" ", firmware_dir=" ", firmware_name=" ", fpga_chip_id=0, ):
    _mode = {'fp32': bs_client.Backend.EMULATOR_FLOAT,
             'quantized': bs_client.Backend.EMULATOR_QUANTIZED,
             'FPGA': bs_client.Backend.FPGA}[mode]
    directory_path = Path(firmware_dir)
    bs_sess = bs_client.Session(_mode, Path(sku_dir), giano_instance_id=0, fpga_chip_id=fpga_chip_id, fpga_request_timeout=600.0)
    bs_sess.start_session()
    if mode == 'FPGA':
        print("FPGA Session started,id: ", fpga_chip_id)
        bs_sess.load_firmware(firmware_directory=directory_path,
                              firmware_name=firmware_name)
    else :
        print("Emulator Session started,mode:", mode)
        bs_sess.load_emulator_firmware(firmware_directory=directory_path,
                                           firmware_name=firmware_name)
    global _native_dim
    _native_dim = bs_sess.parameters.NATIVE_DIM
    print("Firmware loaded successfully")
    return bs_sess

@tvm.register_func("bs.linear",override=True)
def bs_linear(x: tvm.nd.NDArray,
              w: tvm.nd.NDArray,
              b: tvm.nd.NDArray,
              out: tvm.nd.NDArray):
    ############################
    #create fpga session
    ############################
    bs_sess = create_fpga_session(mode='fp32', sku_dir=SKU_PATH, firmware_dir=FIRMWARE_PATH, firmware_name="BERT")

    ############################
    #init linear
    ############################
    # checke size
    num_vec = x.shape[0]
    num_col = x.shape[1]
    num_row = w.shape[0]

    assert w.shape[1] == num_col
    assert b.shape[0] == num_row

    x = x.numpy()
    w = w.numpy()
    b = b.numpy()
    z = torch.zeros(num_vec, num_row, dtype=torch.float32)
    out_torch = torch.from_dlpack(out)

    # set the last element in MVM_IVRF to zero-vector
    bs_sess.load_vector(np.zeros(bs_sess.parameters.NATIVE_DIM), Mem.MvmInitialVrf, bs_sess.parameters.MVM_INITIAL_VRF_SIZE - 1)

    # SLU-specific matrix loading
    if (bs_sess.parameters.ENABLE_FIXED_FUNCTION_SLU):
        i_mat = np.identity(bs_sess.parameters.NATIVE_DIM)
        bs_sess.load_matrix(i_mat, address = 0, memory = Mem.MatrixRf)

    args = Bert.FullyConnectedParams(
            rows = num_row,
            cols = num_col,
            bias = True,
            gelu = True,
            relu = False,
            vecs = num_vec,
            x_startaddr = 0,
            w_startaddr = 1,
            b_startaddr = 0,
            imat_addr = 0,
            asvrf1_scratchpad = 0,
            mfu_scratchpad = 0,
            use_dram = True,
            weight_dram_addr = 0,
            bias_dram_addr = 0,
    )
    # load bias and weight
    bs_sess.load_vector(b, address = args.bias_dram_addr, memory = Mem.Dram)
    bs_sess.load_matrix(w, address = args.weight_dram_addr, memory = Mem.Dram)
    ############################
    #run linear
    ############################
    bs_res = bs_sess.run(Bert.FullyConnectedLayer(args, inputVector = x))  # execute matrix to vector multiplications
    torch.add(torch.asarray(bs_res["outputVector"]), z, out=out_torch)

@tvm.register_func("torch.linear", override=True)
def torch_linear(x: tvm.nd.NDArray,
                 w: tvm.nd.NDArray,
                 b: tvm.nd.NDArray,
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)
    

@tvm.script.ir_module
class test_module:
    @R.function
    def main(x: R.Tensor(("v", "c"), "float32"),
             w: R.Tensor(("r", "c"), "float32"),
             b: R.Tensor(("r", ), "float32")):
        v, c, r= T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("bs.linear", (x, w, b), R.Tensor((v, r), "float32"))
            # lv0 = R.call_dps_packed("torch.linear", (x, w, b), R.Tensor((v, r), "float32"))
            R.output(lv0)
        return lv0

num_vec = 1
num_col = 32
num_row = 16
torch.manual_seed(0)


x = torch.randn((num_vec, num_col), dtype=torch.float32)
w = torch.randn((num_row, num_col), dtype=torch.float32)
b = torch.randn(num_row, dtype=torch.float32)
x_tvm = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(x))
w_tvm = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(w))
b_tvm = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(b))

ex = relax.build(test_module, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](x_tvm,w_tvm,b_tvm)
r0 = torch.mm(x, w.T)
r1 = torch.add(r0, b)
r2 = torch.nn.functional.gelu(r1)
print(r2)
print(nd_res)

