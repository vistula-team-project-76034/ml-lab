import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms

RESULTS = Path("lab03/results")
RESULTS.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# step 1
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full_train = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test_set = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
train_set, val_set = random_split(full_train, [55000, 5000], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128)
test_loader = DataLoader(test_set, batch_size=128)
print(f"train {len(train_set)}, val {len(val_set)}, test {len(test_set)}, device {device}")


# step 2
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49, 64)
        self.pos = nn.Parameter(torch.zeros(1, 16, 64))
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        b = x.shape[0]
        x = x.unfold(2, 7, 7).unfold(3, 7, 7).reshape(b, 16, 49)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


models = {"MLP": MLP(), "CNN": CNN(), "Attention": Attention()}

# step 3
hand_flops = {
    "MLP": {
        "Linear 784->256": 2 * 784 * 256,
        "Linear 256->128": 2 * 256 * 128,
        "Linear 128->10": 2 * 128 * 10,
    },
    "CNN": {
        "Conv 1->16 (28x28)": 2 * 3 * 3 * 1 * 16 * 28 * 28,
        "Conv 16->32 (14x14)": 2 * 3 * 3 * 16 * 32 * 14 * 14,
        "Linear 1568->10": 2 * 1568 * 10,
    },
    "Attention": {
        "Patch proj 49->64": 2 * 16 * 49 * 64,
        "QKV proj x2": 2 * (2 * 16 * 64 * 3 * 64),
        "Scores QK^T x2": 2 * (2 * 16 * 16 * 64),
        "Scores*V x2": 2 * (2 * 16 * 16 * 64),
        "Out proj x2": 2 * (2 * 16 * 64 * 64),
        "FFN x2": 2 * (2 * 16 * 64 * 128 + 2 * 16 * 128 * 64),
        "Head 64->10": 2 * 64 * 10,
    },
}


def latency_ms(model, batch, runs):
    model = model.to("cpu").eval()
    x = torch.randn(batch, 1, 28, 28)
    times = []
    with torch.no_grad():
        for _ in range(10):
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


cost = {}
for name, model in models.items():
    info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    macs, _ = profile(model.to("cpu").eval(), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    hand = sum(hand_flops[name].values())
    cost[name] = {
        "params": info.total_params,
        "hand_flops": hand,
        "thop_macs": int(macs),
        "thop_flops": int(2 * macs),
        "torchinfo_macs": int(info.total_mult_adds),
        "latency_b1_ms": latency_ms(model, 1, 100),
        "latency_b128_ms": latency_ms(model, 128, 20),
    }
    print(name, cost[name])
    print(info)


# step 4
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


history = {}
for name, model in models.items():
    torch.manual_seed(42)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history[name] = []
    t0 = time.perf_counter()
    for epoch in range(5):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        val_acc = evaluate(model, val_loader)
        history[name].append({"epoch": epoch + 1, "loss": total / len(train_set), "val_acc": val_acc})
        print(f"{name} epoch {epoch + 1} loss {total / len(train_set):.4f} val_acc {val_acc:.4f}")
    cost[name]["train_time_s"] = time.perf_counter() - t0
    cost[name]["test_acc"] = evaluate(model, test_loader)
    print(f"{name} test_acc {cost[name]['test_acc']:.4f}")

# step 5
names = list(models)
colors = {"MLP": "tab:blue", "CNN": "tab:orange", "Attention": "tab:green"}

plt.figure(figsize=(6, 4))
for n in names:
    plt.scatter(cost[n]["params"], cost[n]["test_acc"] * 100, s=80, color=colors[n], label=n)
plt.xlabel("Parameters")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs parameters")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "acc_vs_params.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, key, title in [(axes[0], "latency_b1_ms", "Batch 1"), (axes[1], "latency_b128_ms", "Batch 128")]:
    for n in names:
        ax.scatter(cost[n][key], cost[n]["test_acc"] * 100, s=80, color=colors[n], label=n)
    ax.set_xlabel("Median latency (ms, CPU)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Accuracy vs latency, {title}")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "acc_vs_latency.png", dpi=150)

lines = ["Hand-calculated FLOPs per layer (batch 1)"]
for n in names:
    for layer, f in hand_flops[n].items():
        lines.append(f"  {n:10s} {layer:22s} {f:>12,}")
lines.append("")
lines.append(f"{'model':10s} {'params':>9s} {'hand FLOPs':>12s} {'thop MACs':>11s} {'thop x2':>11s} {'torchinfo MACs':>15s} {'lat b1 ms':>10s} {'lat b128 ms':>12s} {'test acc':>9s}")
for n in names:
    c = cost[n]
    lines.append(f"{n:10s} {c['params']:>9,} {c['hand_flops']:>12,} {c['thop_macs']:>11,} {c['thop_flops']:>11,} {c['torchinfo_macs']:>15,} {c['latency_b1_ms']:>10.3f} {c['latency_b128_ms']:>12.3f} {c['test_acc']*100:>8.2f}%")
lines.append("")
lines.append(f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}, train device {device}")
(RESULTS / "results.txt").write_text("\n".join(lines))
(RESULTS / "results.json").write_text(json.dumps({"cost": cost, "history": history}, indent=2))
print("\n".join(lines))
print("Done")
