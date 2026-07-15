# scratch_py — self-made autograd on NumPy × mini-YOLOv5 (Python)

*[日本語](README.md) | English*

A **Python port of `scratch/`** (the zero-dependency C++ version). No deep-learning
framework (PyTorch / TensorFlow) at all — just **NumPy array arithmetic** underneath
a hand-written automatic-differentiation engine that trains a YOLO-style detector.
A teaching artifact for *understanding the training algorithm by building it*, and
readable side-by-side with the C++ version line for line.

## Contents

| File | What |
|------|------|
| `autograd.py` | **The self-made autograd engine**. Tensor + computation graph + backprop.<br>ops: add/sub/mul/matmul/sum/mean, conv2d/maxpool2d/upsample/**batchnorm2d**, cat/slice, maximum/minimum/divide/sqrt/atan/clamp_min, relu/sigmoid/silu |
| `net.py` | mini-YOLOv5 (**C3/SPPF backbone + 2-scale FPN head**), synthetic/disk data, **CIoU loss**, NMS, training loop, image I/O (PPM read, PNG write) |
| `main.py` | no args = numeric gradient check / `train` = training demo |

**Zero dependencies beyond NumPy** (Pillow only optionally, to read JPG/PNG; with
PPM it is pure NumPy).

## Run

```bash
cd scratch_py
python main.py                    # gradient check (proves each backward numerically)
python main.py train              # train -> detect (in-memory synthetic, nc=3, 64px)
python main.py train DIR          # train -> detect (on-disk dataset, nc=3, 64px)
python main.py train DIR 2 128    # train DIR with nc=2 at 128px
```

After training, detections are written to `viz_0.png` … (green = detection, red = GT).

### Sharing data

Labels use the same YOLO format (`class cx cy w h`) as the LibTorch and C++ scratch
versions, so an `images/` + `labels/` folder is **shared across all versions** (data
only; weights are architecture-specific). PPM is read in pure Python (NumPy-only path);
JPG/PNG work if Pillow is installed.

```bash
# e.g. extract person/car from coco128 (tool lives in the parent dir)
python tools/prep_coco_pb.py --src data/coco128 --out data/coco_pb
python scratch_py/main.py train data/coco_pb 2 128
```

## How it works (the learning points)

### 1. Automatic differentiation
Each `Tensor` holds its value `data` (a NumPy array), its gradient `grad`, the
backward closure `backward_fn` that produced it, and its `parents`. The forward pass
builds the computation graph; `loss.backward()` walks it in **reverse-topological
order**, calling each op's `backward_fn` to accumulate gradients into parents via the
chain rule — the same principle as PyTorch's autograd, minimal.

**NumPy only does the array arithmetic**; every derivative (each op's backward) is
written out by hand. What the C++ version wrote as naive nested loops becomes
vectorised NumPy here (e.g. conv = `im2col` then a matmul via `np.einsum`).

### 2. Correctness guarantee (gradient check)
Every hand-written backward is compared against **central finite differences**. Run
`python main.py`: every op comes out at ~1e-7 relative error — numeric proof the
implementation is correct.

> The C++ version is float32 (for SIMD); this one is **float64**. NumPy's natural
> precision keeps the gradient checks crisp instead of drowning in float rounding —
> which is the whole point of this file.

### 3. mini-YOLOv5 (anchors, CIoU, C3/SPPF)
A configuration genuinely shaped like YOLOv5:
- **Anchors**: 3 per scale, ratio matching (< 4), v5-style decode
  (xy = 2σ-0.5+grid, wh = (2σ)²·anchor). Head is field-major (ch = field·na + a).
- **CIoU loss**: `maximum/minimum/divide/sqrt/atan/clamp_min` are hand-made (each
  gradient-checked) and composed into a differentiable CIoU.
- **C3 / SPPF backbone**: stride-2 conv downsampling + C3 (CSP) + SPPF + FPN — a
  v5-equivalent structure.

### 4. BatchNorm and multi-scale (FPN)
- **BatchNorm2d** is hand-made too (forward statistics + the non-trivial backward);
  batch stats while training, running stats at inference. Gradient-checked.
- **2-scale detection**: stride 8 (small objects) and stride 16 (large). The stride-16
  feature is upsampled and fused into stride-8 via an **FPN top-down** path. Each GT is
  **routed to one scale by size** (the YOLO anchor-matching idea).

### 5. YOLO-style loss
Each grid cell predicts `[tx,ty,tw,th, obj, cls...]`:
- box: CIoU regression on object cells only
- obj: objectness, with **positives/negatives normalised separately** to avoid the
  foreground/background imbalance, and the target set to a detached CIoU so confidence
  reflects box quality (fewer false positives)
- cls: class regression on object cells only

### 6. Training
Hand-written SGD with momentum + **linear warmup + cosine decay**. Duplicate boxes are
removed with **NMS** at inference.

## Differences from the C++ scratch version

| | C++ scratch (`../scratch`) | Python (here) |
|---|---|---|
| deps | none (stdlib only) | **NumPy only** (pure Python for PPM) |
| precision | float32 | float64 |
| conv impl | hand im2col+GEMM (+AVX2/threads) | im2col + `np.einsum` |
| speed | fast (optimised) | NumPy-driven (fine in practice; slower than the tuned C++) |
| goal | zero dependency, taken all the way | **shortest / most readable in Python** |

The algorithm is identical. When one version confuses you, read the other: the "C++
nested loop" and the "vectorised NumPy" line up one to one.
