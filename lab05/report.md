# Lab 5: Frameworks and automatic differentiation

Code: `src/run_lab05.py`. All numbers: `results/summary.md` (full data in `results/results.json`), graphs of the model in `results/graph.txt`, loss curves in `results/moons_loss.png`.

## 1. Value node

`Value` keeps 4 things: `data` (number), `grad` (gradient, starts at 0), `parents` (nodes it was made from) and `backward_rule` (function that pushes `out.grad` to the parents using the local derivative). `op` is only for printing.

## 2. Operations

`+`, `*`, `**k`, `relu` (plus `sigmoid` and `log`, needed for BCE in part 5). Each operation creates a new `Value` with the inputs as parents, so the graph is built during the forward pass, like eager mode in PyTorch.

Local derivatives:
- `a + b`: 1 for both
- `a * b`: b for a, a for b
- `a ** k`: k * a^(k-1)
- `relu(a)`: 1 if a > 0 else 0
- `sigmoid(a)`: s * (1 - s)
- `log(a)`: 1 / a

## 3. Backward

1. Topological sort from the output (iterative DFS, so deep graphs don't hit Python recursion limit).
2. Output grad = 1.
3. Go through nodes in reverse order and call `backward_rule`.
4. All rules use `+=`. If a node is used twice, both paths add to its gradient. Test: `x*x + x` at x = 3 gives 7 (2x + 1), with `=` it would be wrong.

## 4. Check on ((a*b + c)**2).relu().sum()

a, b, c are random vectors of size 6. Engine vs PyTorch autograd: max diff 0 (same float64 math). Engine vs numerical gradient (central difference, h = 1e-6): ~1.7e-9. Both are < 1e-6.

## 5. make_moons, 2 -> 8 -> 1

ReLU in the hidden layer, sigmoid output, BCE loss, full batch gradient descent, lr = 0.5, 100 epochs, 100 points. Both versions start from the same weights. The loss curves are the same (max difference ~1e-16), accuracy 0.90 for both.

Median of 5 runs (Kaggle CPU, PyTorch 2.11): my engine 8.89 s, PyTorch 0.032 s, so PyTorch is ~279x faster. Final loss 0.2088 in both. My engine makes a Python object for every scalar operation and calls a Python function for each one in backward. PyTorch does the whole batch as one matrix operation in C++, so there are only a few graph nodes instead of thousands.

## 6. PyTorch graph, execution modes, tensors

Graph: `torch.fx` shows the forward graph (input -> Linear -> ReLU -> Linear -> Sigmoid). `loss.grad_fn` shows the backward graph: `MeanBackward0 -> SigmoidBackward0 -> AddmmBackward0 -> ReluBackward0 -> ...`, and `AccumulateGrad` for the weights. This is the same thing my engine does, just per tensor and not per number.

Eager vs `torch.jit.trace` vs `torch.compile` (Kaggle CPU, 2 threads), after 10 warm-up runs:
- tiny 2-8-1: eager 0.023 ms, trace 0.021 ms, compile 0.054 ms, first compile call ~10.6 s
- medium MLP 512-1024-1024-10: eager 5.22 ms, trace 5.12 ms, compile 5.36 ms, first compile call ~1.0 s

Trace was a bit faster in both cases (less Python overhead per call). Compile was slower than eager in both cases on this CPU, and the first call cost seconds. For the tiny model the work is so small that the time is just overhead, and compile adds its own guards/wrapper. For the medium model almost all time is in the matmuls, which already use optimized BLAS in eager mode, so fusing GELU saves very little.

Tensors: `x.t()` is a view, same memory, only the stride changes from [4096, 1] to [1, 4096], so it is not contiguous. `x.t().sum(dim=1)` took 2.57 ms vs 1.39 ms on a contiguous tensor (~1.8x slower), because memory is read with a big jump. After `.contiguous()` the sum is fast again (1.41 ms), but the copy itself costs ~45 ms, much more than the sum. So copying is only worth it if the tensor is used many times.

## Note: when compilation helps

Compilation helps when the same model with the same input shapes runs many times (training loop, serving) and the model has enough work: many element-wise ops that can be fused (like GELU after Linear), so fewer kernels and less memory traffic. It also removes Python overhead per op.

In my run it did not help: on CPU, for these small MLPs, compile was slower than eager and the first call took 1-11 s. It does not help for a tiny model or one-off runs: the time is mostly overhead, and the compilation cost takes thousands of runs to pay back (if ever). It also hurts when shapes or control flow change a lot, because it has to recompile. The big matmuls are already optimized libraries in eager mode, so compile can't make them much faster; the real gains are usually on GPU with many fusable element-wise ops. Tracing is cheaper to create, but records only one path, so `if` on data is lost. For debugging, eager is easier.
