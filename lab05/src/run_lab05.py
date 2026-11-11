import json
import math
import os
import random
import statistics
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

OUT = "lab05/results"
os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_default_dtype(torch.float64)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# step 1
class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.op = op
        self.backward_rule = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op={self.op!r})"

    # step 2
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def rule():
            self.grad += out.grad
            other.grad += out.grad
        out.backward_rule = rule
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def rule():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward_rule = rule
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def rule():
            self.grad += k * self.data ** (k - 1) * out.grad
        out.backward_rule = rule
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def rule():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out.backward_rule = rule
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def rule():
            self.grad += s * (1 - s) * out.grad
        out.backward_rule = rule
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def rule():
            self.grad += (1.0 / self.data) * out.grad
        out.backward_rule = rule
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    __radd__ = __add__
    __rmul__ = __mul__

    # step 3
    def backward(self):
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                for p in node.parents:
                    stack.append((p, False))
        self.grad = 1.0
        for node in reversed(order):
            node.backward_rule()


def vsum(values):
    total = Value(0.0)
    for v in values:
        total = total + v
    return total


def check_accumulation():
    x = Value(3.0)
    y = x * x + x
    y.backward()
    return x.grad


# step 4
def expr_engine(a, b, c):
    return vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])


def expr_numpy(a, b, c):
    return np.maximum((a * b + c) ** 2, 0).sum()


def numerical_grad(f, arrays, h=1e-6):
    grads = []
    for i in range(len(arrays)):
        g = np.zeros_like(arrays[i])
        for j in range(arrays[i].size):
            plus = [x.copy() for x in arrays]
            minus = [x.copy() for x in arrays]
            plus[i][j] += h
            minus[i][j] -= h
            g[j] = (f(*plus) - f(*minus)) / (2 * h)
        grads.append(g)
    return grads


def verify(n=6):
    a_np, b_np, c_np = [np.random.randn(n) for _ in range(3)]

    vals = [[Value(x) for x in arr] for arr in (a_np, b_np, c_np)]
    out = expr_engine(*vals)
    out.backward()
    eng = [np.array([v.grad for v in vs]) for vs in vals]

    ts = [torch.tensor(arr, requires_grad=True) for arr in (a_np, b_np, c_np)]
    t_out = ((ts[0] * ts[1] + ts[2]) ** 2).relu().sum()
    t_out.backward()
    tor = [t.grad.numpy() for t in ts]

    num = numerical_grad(expr_numpy, [a_np, b_np, c_np])

    return {
        "output_engine": out.data,
        "output_torch": t_out.item(),
        "max_diff_engine_vs_torch": max(float(np.abs(e - t).max()) for e, t in zip(eng, tor)),
        "max_diff_engine_vs_numerical": max(float(np.abs(e - m).max()) for e, m in zip(eng, num)),
        "max_diff_torch_vs_numerical": max(float(np.abs(t - m).max()) for t, m in zip(tor, num)),
    }


# step 5
def init_params():
    w1 = np.random.randn(8, 2) * 0.5
    b1 = np.zeros(8)
    w2 = np.random.randn(1, 8) * 0.5
    b2 = np.zeros(1)
    return w1, b1, w2, b2


def train_engine(X, y, params, epochs, lr):
    w1 = [[Value(x) for x in row] for row in params[0]]
    b1 = [Value(x) for x in params[1]]
    w2 = [[Value(x) for x in row] for row in params[2]]
    b2 = [Value(x) for x in params[3]]
    all_params = [p for row in w1 for p in row] + b1 + [p for row in w2 for p in row] + b2
    losses = []
    for _ in range(epochs):
        terms = []
        for xi, yi in zip(X, y):
            h = [vsum([w * x for w, x in zip(row, xi)]) + b for row, b in zip(w1, b1)]
            h = [v.relu() for v in h]
            z = vsum([w * v for w, v in zip(w2[0], h)]) + b2[0]
            p = z.sigmoid()
            terms.append(-(yi * p.log() + (1 - yi) * (1 - p).log()))
        loss = vsum(terms) / len(X)
        for p in all_params:
            p.grad = 0.0
        loss.backward()
        for p in all_params:
            p.data -= lr * p.grad
        losses.append(loss.data)
    return losses, (w1, b1, w2, b2)


def predict_engine(model, X):
    w1, b1, w2, b2 = model
    preds = []
    for xi in X:
        h = [max(0.0, sum(w.data * x for w, x in zip(row, xi)) + b.data) for row, b in zip(w1, b1)]
        z = sum(w.data * v for w, v in zip(w2[0], h)) + b2[0].data
        preds.append(1 if z > 0 else 0)
    return np.array(preds)


def make_torch_model(params=None):
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
    if params is not None:
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor(params[0]))
            model[0].bias.copy_(torch.tensor(params[1]))
            model[2].weight.copy_(torch.tensor(params[2]))
            model[2].bias.copy_(torch.tensor(params[3]))
    return model


def train_torch(X, y, params, epochs, lr):
    model = make_torch_model(params)
    Xt = torch.tensor(X)
    yt = torch.tensor(y).reshape(-1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model


def timed(fn, runs=5):
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), times, result


def moons_experiment(n=100, epochs=100, lr=0.5):
    X, y = make_moons(n_samples=n, noise=0.1, random_state=0)
    y = y.astype(float)
    params = init_params()

    eng_med, eng_all, (eng_losses, eng_model) = timed(lambda: train_engine(X, y, params, epochs, lr))
    tor_med, tor_all, (tor_losses, tor_model) = timed(lambda: train_torch(X, y, params, epochs, lr))

    eng_acc = float((predict_engine(eng_model, X) == y).mean())
    with torch.no_grad():
        tor_acc = float(((tor_model(torch.tensor(X)) > 0.5).numpy().ravel() == y).mean())

    plt.figure(figsize=(6, 4))
    plt.plot(eng_losses, label="my engine")
    plt.plot(tor_losses, "--", label="PyTorch")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/moons_loss.png", dpi=120)
    plt.close()

    return {
        "n_samples": n,
        "epochs": epochs,
        "lr": lr,
        "engine_median_s": eng_med,
        "torch_median_s": tor_med,
        "engine_runs_s": eng_all,
        "torch_runs_s": tor_all,
        "speedup_torch_vs_engine": eng_med / tor_med,
        "engine_final_loss": eng_losses[-1],
        "torch_final_loss": tor_losses[-1],
        "max_loss_diff": float(np.abs(np.array(eng_losses) - np.array(tor_losses)).max()),
        "engine_accuracy": eng_acc,
        "torch_accuracy": tor_acc,
    }


# step 6
def autograd_graph(fn, depth=0, lines=None):
    if lines is None:
        lines = []
    if fn is None:
        return lines
    lines.append("  " * depth + type(fn).__name__)
    for child, _ in fn.next_functions:
        autograd_graph(child, depth + 1, lines)
    return lines


def inspect_graph():
    model = make_torch_model()
    x = torch.randn(4, 2)
    loss = model(x).mean()
    fx_graph = str(torch.fx.symbolic_trace(model).graph)
    grad_graph = "\n".join(autograd_graph(loss.grad_fn))
    with open(f"{OUT}/graph.txt", "w") as f:
        f.write("torch.fx graph (forward)\n")
        f.write(fx_graph + "\n\n")
        f.write("autograd graph (backward, from loss)\n")
        f.write(grad_graph + "\n")
    return fx_graph, grad_graph


def bench(fn, x, warmup=10, runs=50):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def first_call_ms(fn, x):
    with torch.no_grad():
        start = time.perf_counter()
        fn(x)
    return (time.perf_counter() - start) * 1e3


def execution_modes():
    torch.set_default_dtype(torch.float32)
    cases = {
        "tiny 2-8-1, batch 100": (make_torch_model(), torch.randn(100, 2)),
        "medium 512-1024-1024-10, batch 256": (
            nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 10)),
            torch.randn(256, 512),
        ),
    }
    rows = []
    for name, (model, x) in cases.items():
        model.eval()
        row = {"model": name, "eager_ms": bench(model, x)}
        with torch.no_grad():
            traced = torch.jit.trace(model, x)
        row["trace_first_call_ms"] = first_call_ms(traced, x)
        row["trace_ms"] = bench(traced, x)
        try:
            compiled = torch.compile(model)
            row["compile_first_call_ms"] = first_call_ms(compiled, x)
            row["compile_ms"] = bench(compiled, x)
        except Exception as e:
            row["compile_error"] = type(e).__name__ + ": " + str(e).splitlines()[0][:150]
        rows.append(row)
    torch.set_default_dtype(torch.float64)
    return rows


def tensor_info(name, t):
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "stride": list(t.stride()),
        "contiguous": t.is_contiguous(),
    }


def contiguity():
    x = torch.randn(4096, 4096, dtype=torch.float32)
    xt = x.t()
    xt_copy = xt.contiguous()
    model = make_torch_model()
    info = [
        tensor_info("x", x),
        tensor_info("x.t() (view)", xt),
        tensor_info("x.t().contiguous()", xt_copy),
        tensor_info("Linear(2,8).weight", model[0].weight),
        tensor_info("Linear(2,8).weight.t()", model[0].weight.t()),
    ]
    timing = {
        "sum_dim1_contiguous_ms": bench(lambda t: t.sum(dim=1), x),
        "sum_dim1_noncontiguous_view_ms": bench(lambda t: t.sum(dim=1), xt),
        "sum_dim1_after_contiguous_copy_ms": bench(lambda t: t.sum(dim=1), xt_copy),
        "contiguous_copy_ms": bench(lambda t: t.contiguous(), xt),
        "shares_memory_view": xt.data_ptr() == x.data_ptr(),
    }
    timing["slowdown_noncontiguous"] = timing["sum_dim1_noncontiguous_view_ms"] / timing["sum_dim1_contiguous_ms"]
    return info, timing


def write_summary(res):
    v = res["verification"]
    m = res["moons"]
    lines = [
        "# Lab 5 results",
        "",
        f"PyTorch {res['torch_version']}, threads: {res['threads']}",
        "",
        "## Gradient check: ((a*b + c)**2).relu().sum()",
        "",
        "| comparison | max abs diff |",
        "|---|---|",
        f"| engine vs PyTorch | {v['max_diff_engine_vs_torch']:.2e} |",
        f"| engine vs numerical | {v['max_diff_engine_vs_numerical']:.2e} |",
        f"| PyTorch vs numerical | {v['max_diff_torch_vs_numerical']:.2e} |",
        "",
        f"Accumulation test x*x + x at x=3: grad = {res['accumulation_grad']} (expected 7.0)",
        "",
        f"## make_moons, 2-8-1, {m['n_samples']} samples, {m['epochs']} epochs, lr={m['lr']} (median of 5 runs)",
        "",
        "| | median time, s | final loss | accuracy |",
        "|---|---|---|---|",
        f"| my engine | {m['engine_median_s']:.3f} | {m['engine_final_loss']:.4f} | {m['engine_accuracy']:.2f} |",
        f"| PyTorch | {m['torch_median_s']:.3f} | {m['torch_final_loss']:.4f} | {m['torch_accuracy']:.2f} |",
        "",
        f"PyTorch is {m['speedup_torch_vs_engine']:.0f}x faster. Max loss difference over all epochs: {m['max_loss_diff']:.2e}",
        "",
        "## Eager vs traced vs compiled (inference, median of 50 runs after 10 warm-up)",
        "",
        "| model | eager, ms | trace, ms | compile, ms | trace 1st call, ms | compile 1st call, ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["execution_modes"]:
        comp = f"{r['compile_ms']:.3f}" if "compile_ms" in r else "n/a"
        comp1 = f"{r['compile_first_call_ms']:.0f}" if "compile_first_call_ms" in r else "n/a"
        lines.append(
            f"| {r['model']} | {r['eager_ms']:.3f} | {r['trace_ms']:.3f} | {comp} | {r['trace_first_call_ms']:.1f} | {comp1} |"
        )
    for r in res["execution_modes"]:
        if "compile_error" in r:
            lines.append(f"\ntorch.compile failed for {r['model']}: {r['compile_error']}")
    lines += [
        "",
        "## Tensor shape, dtype, contiguity",
        "",
        "| tensor | shape | dtype | stride | contiguous |",
        "|---|---|---|---|---|",
    ]
    for t in res["tensors"]:
        lines.append(f"| {t['name']} | {t['shape']} | {t['dtype']} | {t['stride']} | {t['contiguous']} |")
    c = res["contiguity_timing"]
    lines += [
        "",
        "| operation on 4096x4096 float32 | ms |",
        "|---|---|",
        f"| x.sum(dim=1), contiguous | {c['sum_dim1_contiguous_ms']:.3f} |",
        f"| x.t().sum(dim=1), non-contiguous view | {c['sum_dim1_noncontiguous_view_ms']:.3f} |",
        f"| x.t().contiguous().sum(dim=1) | {c['sum_dim1_after_contiguous_copy_ms']:.3f} |",
        f"| x.t().contiguous() copy itself | {c['contiguous_copy_ms']:.3f} |",
        "",
        f"Non-contiguous view is {c['slowdown_noncontiguous']:.1f}x slower. View shares memory with x: {c['shares_memory_view']}",
        "",
    ]
    with open(f"{OUT}/summary.md", "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def main():
    res = {"torch_version": torch.__version__, "threads": torch.get_num_threads()}
    res["accumulation_grad"] = check_accumulation()
    res["verification"] = verify()
    print("verification:", res["verification"])
    res["moons"] = moons_experiment()
    print("moons:", {k: v for k, v in res["moons"].items() if not k.endswith("runs_s")})
    fx_graph, grad_graph = inspect_graph()
    print(fx_graph)
    print(grad_graph)
    res["execution_modes"] = execution_modes()
    res["tensors"], res["contiguity_timing"] = contiguity()
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(write_summary(res))
    assert res["verification"]["max_diff_engine_vs_torch"] < 1e-6
    assert res["verification"]["max_diff_engine_vs_numerical"] < 1e-6


if __name__ == "__main__":
    main()
