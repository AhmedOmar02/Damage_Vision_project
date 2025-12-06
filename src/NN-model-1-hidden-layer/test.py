import cupy as cp
x = cp.arange(10)
print(x.device)  # should print: cuda:0
