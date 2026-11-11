# Lab 5 results

PyTorch 2.11.0+cpu, threads: 2

## Gradient check: ((a*b + c)**2).relu().sum()

| comparison | max abs diff |
|---|---|
| engine vs PyTorch | 0.00e+00 |
| engine vs numerical | 1.68e-09 |
| PyTorch vs numerical | 1.68e-09 |

Accumulation test x*x + x at x=3: grad = 7.0 (expected 7.0)

## make_moons, 2-8-1, 100 samples, 100 epochs, lr=0.5 (median of 5 runs)

| | median time, s | final loss | accuracy |
|---|---|---|---|
| my engine | 8.890 | 0.2088 | 0.90 |
| PyTorch | 0.032 | 0.2088 | 0.90 |

PyTorch is 279x faster. Max loss difference over all epochs: 3.33e-16

## Eager vs traced vs compiled (inference, median of 50 runs after 10 warm-up)

| model | eager, ms | trace, ms | compile, ms | trace 1st call, ms | compile 1st call, ms |
|---|---|---|---|---|---|
| tiny 2-8-1, batch 100 | 0.023 | 0.021 | 0.054 | 0.9 | 10551 |
| medium 512-1024-1024-10, batch 256 | 5.215 | 5.121 | 5.356 | 6.1 | 978 |

## Tensor shape, dtype, contiguity

| tensor | shape | dtype | stride | contiguous |
|---|---|---|---|---|
| x | [4096, 4096] | torch.float32 | [4096, 1] | True |
| x.t() (view) | [4096, 4096] | torch.float32 | [1, 4096] | False |
| x.t().contiguous() | [4096, 4096] | torch.float32 | [4096, 1] | True |
| Linear(2,8).weight | [8, 2] | torch.float64 | [2, 1] | True |
| Linear(2,8).weight.t() | [2, 8] | torch.float64 | [1, 2] | False |

| operation on 4096x4096 float32 | ms |
|---|---|
| x.sum(dim=1), contiguous | 1.387 |
| x.t().sum(dim=1), non-contiguous view | 2.566 |
| x.t().contiguous().sum(dim=1) | 1.407 |
| x.t().contiguous() copy itself | 44.537 |

Non-contiguous view is 1.8x slower. View shares memory with x: True
