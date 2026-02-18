# test_gpu.py

import time

import cupy as cp
import numpy as np
import torch

# Test PyTorch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

# Quick benchmark: matrix multiply on CPU vs GPU
size = 5000
a = torch.randn(size, size)
start = time.time()
c = a @ a
print(f"CPU: {time.time() - start:.3f}s")

a_gpu = a.cuda()
torch.cuda.synchronize()
start = time.time()
c_gpu = a_gpu @ a_gpu
torch.cuda.synchronize()
print(f"GPU: {time.time() - start:.3f}s")

# Test CuPy (drop-in numpy replacement)
size = 5000
a_np = np.random.randn(size, size).astype(np.float32)
a_cp = cp.asarray(a_np)

start = time.time()
np.dot(a_np, a_np)
print(f"NumPy (CPU): {time.time() - start:.3f}s")

start = time.time()
cp.dot(a_cp, a_cp)
cp.cuda.Stream.null.synchronize()
print(f"CuPy (GPU): {time.time() - start:.3f}s")
