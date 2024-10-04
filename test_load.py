import triton.language as tl
import triton
import torch


@triton.jit
def test_fn(out_ptr, a_ptr, desc, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    block = tl._experimental_descriptor_load(desc, [0, 0], [M_BLOCK, N_BLOCK], a_ptr.dtype.element_ty)
    tl.store(out_ptr + tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :], block)


inp = torch.arange(32 * 128, dtype=torch.float16).reshape(32, 128).cuda()

M_BLOCK = 4
N_BLOCK = 16
out = torch.empty((M_BLOCK, N_BLOCK), dtype=torch.float16, device="cuda")

from triton.tools.experimental_descriptor import create_2d_tma_descriptor

cpu_desc = create_2d_tma_descriptor(inp.data_ptr(), *inp.shape, M_BLOCK, N_BLOCK, inp.element_size())

workspace = torch.zeros(128, dtype=torch.uint8).cuda()
print(inp[:M_BLOCK, :N_BLOCK])
test_fn[(1, )](out, inp, cpu_desc, M_BLOCK, N_BLOCK)
print(out.cpu())
