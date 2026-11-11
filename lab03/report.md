# Lab 3: Architectures and their costs

Run: `python lab03/src/lab03.py` from the repo root. Outputs are in `lab03/results/`
(`results.txt`, `results.json`, `acc_vs_params.png`, `acc_vs_latency.png`).

## 1. Setup

- Fashion-MNIST, `ToTensor()` + `Normalize((0.2860,), (0.3530,))`
- official train 60k split into 55k train / 5k validation (seed 42), official 10k test
- batch size 128, Adam lr 1e-3, 5 epochs, CrossEntropyLoss
- latency: CPU, `model.eval()` + `torch.no_grad()`, 10 warm-up runs, then median of 100 runs (batch 1) and 20 runs (batch 128)

Models (as in the manual):

- **MLP**: 784 → 256 → 128 → 10 with ReLU
- **CNN**: Conv(1→16, 3×3) → ReLU → MaxPool → Conv(16→32, 3×3) → ReLU → MaxPool → Linear(1568→10)
- **Attention**: 16 patches of 7×7 → Linear(49→64) + learned position embedding → 2 × TransformerEncoderLayer(d_model=64, nhead=4, ff=128) → mean pool → Linear(64→10)

I added a position embedding (16×64 = 1,024 params) to the attention model. Without it the model
does not know where each patch is, because mean pooling ignores the order of the patches.

Parameter count (torchinfo): MLP 235,146, CNN 20,490, Attention 71,818. The manual says "target
100k–150k", but with the exact layers from the manual the sizes come out different, so I kept the
manual's layers and just report the real numbers.

## 2. Hand-calculated FLOPs (batch 1)

Formulas from Ch. 4: Linear = 2 × in × out, Conv2d = 2 × K × K × C_in × C_out × H_out × W_out,
attention per layer: QKV = 2 × seq × d × 3d, scores = 2 × seq² × d, scores·V = 2 × seq² × d,
out proj = 2 × seq × d × d, FFN = 2 × seq × d × ff × 2. Here seq = 16, d = 64, ff = 128.

| Model | Layer | Calculation | FLOPs |
|---|---|---|---|
| MLP | Linear 784→256 | 2·784·256 | 401,408 |
| MLP | Linear 256→128 | 2·256·128 | 65,536 |
| MLP | Linear 128→10 | 2·128·10 | 2,560 |
| CNN | Conv 1→16 | 2·3·3·1·16·28·28 | 225,792 |
| CNN | Conv 16→32 | 2·3·3·16·32·14·14 | 1,806,336 |
| CNN | Linear 1568→10 | 2·1568·10 | 31,360 |
| Attn | Patch proj | 2·16·49·64 | 100,352 |
| Attn | QKV (one layer) | 2·16·64·192 | 393,216 |
| Attn | QKᵀ (one layer) | 2·16²·64 | 32,768 |
| Attn | scores·V (one layer) | 2·16²·64 | 32,768 |
| Attn | out proj (one layer) | 2·16·64·64 | 131,072 |
| Attn | FFN (one layer) | 2·16·64·128·2 | 524,288 |
| Attn | head | 2·64·10 | 1,280 |

One encoder layer = 1,114,112 FLOPs, two layers = 2,228,224.
Totals: **MLP 469,504, CNN 2,063,488, Attention 2,329,856**.

## 3. Hand vs measured

| Model | Params | Hand FLOPs | thop MACs | thop × 2 | torchinfo MACs |
|---|---|---|---|---|---|
| MLP | 235,146 | 469,504 | 234,752 | 469,504 | 235,146 |
| CNN | 20,490 | 2,063,488 | 1,031,744 | 2,063,488 | 1,050,570 |
| Attention | 71,818 | 2,329,856 | 591,488 | 1,182,976 | 37,514 |

Explaining the gaps:

1. **Factor 2.** thop returns MACs (multiply-accumulate). One MAC = 2 FLOPs, so thop × 2 is what
   should match the hand numbers. For MLP and CNN it matches exactly.
2. **Bias / ReLU / pooling.** Hand formulas ignore bias adds, ReLU and MaxPool. thop also ignores
   them here, torchinfo adds the bias for conv (that is why its CNN number is 1,050,570 and not
   1,031,744). These are small (<2%).
3. **Attention is undercounted by thop (about 2×).** thop has no rule for `nn.MultiheadAttention`
   (it prints "Cannot find rule ... Treat it as zero Macs"). So QKV projections, QKᵀ, scores·V and
   the out projection are counted as 0. thop only sees the patch projection, the FFN linears, the
   head and LayerNorm: 50,176 + 524,288 + 640 + 16,384 = 591,488 MACs. The missing part is
   (393,216 + 32,768 + 32,768 + 131,072) × 2 layers = 1,179,648 FLOPs, which is the gap.
4. **torchinfo for Attention is way too low (37,514).** torchinfo counts Linear mult-adds as
   "number of weights", so it does not multiply by the 16 tokens, and it also skips the attention
   matmuls. So for sequence models the torchinfo number is not usable as FLOPs.

Conclusion: profilers are fine for standard layers (Linear, Conv), but for attention the formula
from the book gives the real cost and the tool silently misses half of it.

## 4. Latency and accuracy

| Model | Latency b=1 (ms) | Latency b=128 (ms) | Per image at b=128 (ms) | Test acc |
|---|---|---|---|---|
| MLP | 0.065 | 0.51 | 0.004 | 87.93% |
| CNN | 0.222 | 9.46 | 0.074 | 89.60% |
| Attention | 0.436 | 3.24 | 0.025 | 85.89% |

(Kaggle CPU, 2 threads, torch 2.11. Latency numbers will be different on another machine, the order is what matters.)

Validation accuracy after each epoch is in `results.json`.

Plots: `results/acc_vs_params.png`, `results/acc_vs_latency.png`.

## 5. Trade-offs (Ch. 4.6–4.7)

**FLOPs do not predict latency.** CNN and Attention have similar FLOPs (2.1M vs 2.3M), but at batch 128
the CNN is 2.9× slower, and the MLP is 4.4× fewer FLOPs than the CNN but ~19× faster at batch 128.
The reason is the memory access pattern (Table 4.2 in the book):

- **MLP – sequential access, broadcast.** It is just three matrix multiplications. The weight matrix is
  read sequentially and each weight is reused for all 128 images in the batch, so batch 128 costs only
  ~8× batch 1. This is the case CPUs and BLAS are best at. But it has the most parameters
  (235k ≈ 0.9 MB) because every pixel is connected to every neuron, and it ignores that pixels next to
  each other are related, so accuracy is lower than CNN.
- **CNN – strided access, sliding window.** Only 20k parameters because the same 3×3 kernels are
  reused at every position (parameter sharing, §4.6.2). This gives the best accuracy with the fewest
  parameters. But the cost moved from weights to activations: the feature maps (16×28×28 per image)
  are large and the convolution has to read 3×3 windows with a stride, which on CPU means im2col /
  extra memory traffic. With only 1 and 16 input channels the matmuls are small and inefficient, so it is
  the slowest at batch 128.
- **Attention – random access, broadcast + gather.** Most of its FLOPs are dense matmuls (QKV, FFN),
  which run fast at batch 128 (faster than CNN per image). At batch 1 it is the slowest because it is
  many small operations (projections, reshapes for 4 heads, softmax, 2 LayerNorms per layer), so the
  time is overhead, not math. The seq² part is small here (seq = 16: QKᵀ + scores·V are only 5.6% of
  FLOPs), but it would grow quadratically with more patches (Table 4.3: activations O(B × N²)).
  Accuracy is the lowest: with 7×7 patches and 5 epochs the model has no built-in idea of locality and
  needs more data/training to learn it.

**Most cost-effective:** for this task the **CNN**. It has the highest accuracy (89.6%) with 11× fewer
parameters than the MLP, so it is the best choice when memory/model size is the limit (edge, mobile).
If latency on CPU is the only thing that matters, the **MLP** wins (0.065 ms at batch 1, 1.7 pp less accuracy).
The attention model is not cost-effective at this scale: more FLOPs than CNN, slower at batch 1, and
the lowest accuracy.
