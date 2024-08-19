import torch
from kernel import test
from typing import Optional

def pytest(a: torch.Tensor,
           b: torch.Tensor,
           out: Optional[torch.Tensor] = None,
           block_size: int = 128) -> torch.Tensor:
    assert a.shape == b.shape
    if out is None:
        out = torch.empty_like(a)
    assert a.shape == out.shape

    test(a.ravel(), b.ravel(), out.ravel(), block_size)
    return out.reshape(a.shape)

# Create example tensors
a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
b = torch.tensor([5.0, 6.0, 7.0, 8.0], device='cuda')

c = pytest(a, b)

print(c)