# scratch — a from-scratch, zero-dependency YOLO

*English | [日本語](README.md)*

**Using no external libraries whatsoever (standard library only)**, this trains a
YOLO-style object detector with nothing but a self-made automatic-differentiation
engine. No LibTorch, no BLAS, no Eigen. A study project for *fully understanding
the training algorithm with your own hands*.

## Results

![results](../assets/results.png)

Detections from the mini-YOLOv5 trained with the self-made autograd alone
(green = detection, red = ground truth). Trained on coco128 person/car at 128px
for 2500 steps (loss 0.88). Left: person (conf 0.95) / middle: multiple people /
right: car (conf 0.86). It looks coarse because the input is 128px — that's
exactly the resolution the model sees.

## Contents

| File | What it is |
|------|------------|
| `autograd.h/.cpp` | **self-made autograd engine**: tensors + compute graph + backprop.<br>ops: add/sub/mul/matmul/sum/mean, conv2d/maxpool2d/upsample/**batchnorm2d**, cat/slice, relu/sigmoid/silu |
| `net.h/.cpp` | the small YOLO (**Conv+BN+SiLU** blocks + **2-scale FPN head**), synthetic/disk data, YOLO-style loss (size-based scale assignment), NMS, training loop |
| `stb_image*.h` / `stb_impl.cpp` | image IO only (JPG/PNG/BMP). Single-header, public-domain. **Not used by the learning core** (the PPM path stays pure C++) |
| `main.cpp` | `test` = numerical gradient checks / `train` = training demo |

## Build & run

```powershell
cd scratch
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

build/Release/scratch_yolo.exe             # gradient checks (verify hand-written backward vs numeric)
build/Release/scratch_yolo.exe train       # train → detect (in-memory synthetic data)
build/Release/scratch_yolo.exe train DIR   # train → detect (on-disk dataset)
```

### Loading data (JPG/PNG/PPM)

Image IO uses **stb_image / stb_image_write** (single-header, public-domain) so it
can **read JPG / PNG / BMP directly and write PNG visualizations directly**. The
PPM path remains pure C++, so if you want to run with *truly zero dependencies*,
use PPM (**the learning algorithm itself is always standard-library-only**).

Labels use the same YOLO format as the LibTorch version (`class cx cy w h`), so
`images/` + `labels/` folders can be **shared by both versions** (weights can't be
shared — different architectures — but the data can).

```powershell
# e.g. extract person/car from coco128 (copies JPGs + labels)
python tools/prep_coco_pb.py --src data/coco128 --out data/coco_pb
# train the scratch version (reads JPG directly, nc=2, 128px)
scratch/build/Release/scratch_yolo.exe train data/coco_pb 2 128
```

## How it works (the learning points)

### 1. Automatic differentiation (autograd)
Every tensor carries its value `data`, gradient `grad`, the backward closure
`backward_fn` of the op that created it, and its parent nodes (`Node`). The
forward pass builds the compute graph naturally; `loss.backward()` walks the graph
in **reverse topological order**, calls each `backward_fn`, and accumulates
gradients into parents via the chain rule. This is a minimal reproduction of the
same principle behind PyTorch's autograd.

### 2. Correctness guarantee (gradient check)
Each op's analytic gradient (backward) is compared against **central-difference
numerical gradients**. Every op has relative error ~1e-3 or below — the
implementation is numerically proven correct before building on top of it.

### mini-YOLOv5 (①anchors ②CIoU ③C3/SPPF, all dependency-free)
A configuration pushed toward the real YOLOv5 family is also implemented:
- **① anchors**: 3 anchors per scale, ratio matching (< 4), v5-style decode
  (xy = 2σ-0.5+grid, wh = (2σ)²·anchor). The head is field-major (ch = field·na + a).
- **② CIoU loss**: `maximum/minimum/divide/sqrt/atan/clamp_min` are hand-written
  (each gradient-checked) and composed into a differentiable CIoU.
- **③ C3 / SPPF backbone**: stride-2 conv downsampling + C3 (CSP) + SPPF + FPN,
  i.e. a v5-equivalent structure.

Now both the *components* and the *detection mechanism* are YOLOv5-family. Being a
~26-layer net, though, **it needs many steps to train well** (at 500 steps / 128px
it detects large objects cleanly but over-detects crowded/small objects — a matter
of training budget / objectness calibration, not architecture).

### 3. BatchNorm and multi-scale (FPN)
- **BatchNorm2d** is hand-written too (forward statistics + the non-trivial
  backward). It uses batch statistics during training and running statistics at
  inference. The gradient is numerically verified.
- **2-scale detection**: stride 8 (small objects) and stride 16 (large objects),
  with a **top-down FPN** that upsamples the stride-16 feature and fuses it into
  the stride-8 map. Each GT is **routed to a scale by size** (the idea behind
  YOLO's anchor matching).

### 4. YOLO-style loss
Each grid cell at each scale predicts `[tx,ty,tw,th, obj, cls...]`.
- box: object cells only — regress center offset + w,h
- obj: objectness. **positives and negatives are normalized separately** to avoid
  the foreground/background imbalance (important)
- cls: class regression at object cells only

### 5. Training and post-processing
Hand-written SGD with momentum + **cosine decay**. At inference, duplicate boxes
are removed with **NMS**.
> Note (a lesson): once BN makes the net deeper, the learning rate must be lowered
> or it diverges. lr0=0.03 + cosine decay + 500 steps converges stably (loss 7.7→0.05).

### 6. Speed-up (~10× while staying dependency-free)
Convolution is reformulated as **im2col + GEMM**: unfold receptive fields into a
patch matrix and do one matmul `W(O,K) @ col(K,P)` (forward and backward). GEMM
uses `i-p-j` loop order so the inner loop is contiguous, and `/arch:AVX2 /fp:fast`
(pure compiler flags) enable auto-vectorization. On top of that, **conv2d is
multithreaded over the batch** (`std::thread` only): the forward is independent
per image; in the backward, dW/dBias are accumulated into thread-local buffers and
reduced under a lock (dInput is per-image, so it needs no lock).

Cumulative effect (500 steps):

| version | time | /step | vs naive |
|---|---|---|---|
| naive convolution | ~1500s | ~3s | 1× |
| im2col+GEMM+AVX2 | 157.6s | 0.315s | ~10× |
| + multithreading | **86.9s** | **0.173s** | **~17×** |

Correctness re-verified by numerical gradient checks. No external library (BLAS
etc.) is used.
> The threading gain is only ~1.8×: batch 12 gives low parallelism, and the
> non-conv parts (BN/maxpool/loss) are serial (Amdahl's law). A thread pool or
> larger batch would push it further.

## Differences from the LibTorch version

| | LibTorch version (parent dir) | scratch version (here) |
|---|---|---|
| deps | requires LibTorch | **none (standard library only)** |
| scale | full YOLOv5n (1.9M params) | small (study-sized) |
| speed | medium (optimized kernels) | slow (naive loops) |
| goal | practical + learning | **all-in on understanding/learning** |

The naive convolution loops aren't suited to real large-scale training, but you
can follow *why the training works* line by line. That's the value of this
directory.
