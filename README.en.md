# yolov5_cpp

*English | [日本語](README.md)*

**YOLOv5 training + inference in C++ (LibTorch)** — a C++ port of the Ultralytics
YOLOv5 (PyTorch) training pipeline. It's a study project where you can follow the
internals (model, loss, anchor matching, backprop) in C++, and it's practical
enough to actually train on small/medium datasets.

In addition, `scratch/` contains a **mini-YOLOv5 built from scratch with ZERO
external dependencies (standard library only)** — the image below is its output.

![results](assets/results.png)

*The dependency-free, self-made-autograd mini-YOLOv5 detecting person/car on real
photos (green = detection, red = ground truth). See [`scratch/README.en.md`](scratch/README.en.md).*

## Layout

```
src/
  model/modules.h     Conv / Bottleneck / C3 / SPPF   (≈ models/common.py)
  model/yolov5.*      backbone + PAN head + Detect     (YOLOv5n assembled in C++)
  loss/loss.*         build_targets + CIoU + BCE       (ComputeLoss port)
  data/dataset.*      YOLO-format loader + letterbox + augmentation
  data/image_io.*     image IO via stb_image (no external deps)
  utils.*             NMS / xywh2xyxy / box drawing
  train.cpp           training loop (warmup + cosine LR + EMA + checkpoint)
  detect.cpp          inference (decode + NMS + visualization)
  gen_data.cpp        synthetic data generator (for smoke tests)
  main.cpp            CLI
```

The only dependency is **LibTorch 2.5.1 (CPU)**. Image IO uses the bundled
stb_image, so OpenCV is not required (auto-detected by CMake if present).

## Build

```powershell
# Assumes LibTorch is extracted at C:\prog\vc\third_party\libtorch
# (kept outside the project so zipping yolov5_cpp doesn't include it)
powershell -ExecutionPolicy Bypass -File build.ps1
# => build/Release/yolov5.exe
```

> Get LibTorch (CPU 2.5.1) and extract to `C:\prog\vc\third_party\libtorch`:
> `https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip`
> Put the stb headers in `C:\prog\vc\third_party\stb`. For a different location,
> override with `cmake ... -DCMAKE_PREFIX_PATH=<path>/libtorch`.

## Usage

### 1. Smoke test (train → detect on synthetic data)

```powershell
# generate 3-class synthetic data (red/green/blue rectangles)
build/Release/yolov5.exe gen-data --out data/synthetic --num-train 64 --num-val 16 --imgsz 320

# train
build/Release/yolov5.exe train --images data/synthetic/images/train --nc 3 `
    --epochs 30 --batch 8 --imgsz 320 --out runs

# detect (run trained weights, write a visualization PNG)
build/Release/yolov5.exe detect --weights runs/best.pt --nc 3 `
    --source data/synthetic/images/val/img_0000.png --imgsz 320 --out pred.png
```

### 2. Train on your own data

Same layout as Ultralytics (`images/` paired with `labels/`; each label is one
object per line, `class cx cy w h`, normalized 0..1):

```
mydata/
  images/train/*.jpg
  labels/train/*.txt   # same basename as the image
```

```powershell
build/Release/yolov5.exe train --images mydata/images/train --nc <num-classes> --epochs 100 --batch 16 --imgsz 640
```

`--labels DIR` can be given explicitly; otherwise the `images` segment of the
path is replaced with `labels`.

### 3. ONNX export (Python bridge)

LibTorch (C++) has no ONNX exporter, so trained weights are handed to Python for
conversion.

```powershell
# deps (first time): CPU torch + onnx + onnxruntime
python -m pip install torch onnx onnxruntime --index-url https://download.pytorch.org/whl/cpu

# (1) write weights to a portable format (weights.bin) from C++
build/Release/yolov5.exe export-weights --weights runs/best.pt --nc 3 --out weights.bin

# (2) load into a structurally identical PyTorch model and export ONNX
python tools/export_onnx.py --weights weights.bin --nc 3 --imgsz 320 --out model.onnx
```

`tools/export_onnx.py` rebuilds a PyTorch YOLOv5n with the **same layer names** as
`src/model/`, so the `state_dict` keys match exactly (`missing 0 / unexpected 0`).
The exported ONNX returns decoded `(batch, N, nc+5)` and matches the PyTorch
output to within ~2e-4.

### 4. Import official Ultralytics weights (official .pt → run in C++)

You can run the official `yolov5n.pt` in this C++ version. The architecture is an
exact match (all 349 tensors map cleanly), so it's just a layer-name remap. The
yolov5 repo is NOT needed (stub modules let us unpickle the official checkpoint
and extract only the tensors).

```powershell
# fetch official weights
curl -L -o yolov5n.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt

# (1) official .pt -> portable bin (remaps model.N -> conv0/c3_2/...)
python tools/convert_ultralytics.py --pt yolov5n.pt --out weights_official.bin

# (2) bin -> a .pt the C++ side can load
build/Release/yolov5.exe import-weights --in weights_official.bin --nc 80 --out yolov5n_cpp.pt

# (3) detect (COCO 80 classes)
build/Release/yolov5.exe detect --weights yolov5n_cpp.pt --nc 80 --source bus.jpg --imgsz 640 --out pred.png
```

Verified: on `bus.jpg` it correctly detects 4 persons + 1 bus (same as the
official YOLOv5n). The reverse direction (C++ weights → official format) is
possible via `export-weights` + the inverse remap.

## Implementation notes

- **Conv = Conv2d(no bias) → BatchNorm → SiLU**. C3 is the CSP block, SPPF is 3
  chained maxpools.
- **Detect**: per scale, `1x1 Conv → (bs, na, ny, nx, nc+5)`. Strides are computed
  automatically from a dummy forward pass; anchors are converted to grid units.
- **Loss**: `build_targets` replicates each GT into neighbouring cells (±0.5) and
  matches anchors by ratio (< 4). box = CIoU, obj = BCE with IoU as the target,
  cls = BCE.
- **Training**: SGD (momentum=0.937, nesterov) + linear warmup + cosine decay +
  EMA. `best.pt` holds the EMA weights, `last.pt` the raw weights.

## From-scratch version (`scratch/`)

A study-oriented YOLO **built from autograd upward with zero external
dependencies (standard library only)** lives in `scratch/`. No LibTorch — tensors,
the compute graph, backprop, conv/pool, loss, SGD and NMS are all hand-written,
with correctness verified by numerical gradient checks. See
[`scratch/README.en.md`](scratch/README.en.md).

```powershell
cd scratch && cmake -S . -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release
build/Release/scratch_yolo.exe          # gradient checks
build/Release/scratch_yolo.exe train    # train → detect
```

### Python version (`scratch_py/`)

A **NumPy-based port** of the from-scratch version above. No deep-learning framework
(PyTorch etc.) — a hand-written autograd engine (float64) on top of **NumPy array
arithmetic** trains the same mini-YOLOv5, readable side-by-side with the C++ version.
See [`scratch_py/README.en.md`](scratch_py/README.en.md).

```bash
cd scratch_py
python main.py          # gradient checks (every op matches finite diff, ~1e-7)
python main.py train    # train → detect (writes viz_*.png)
```

Zero dependencies beyond NumPy (Pillow only optionally, to read JPG/PNG; pure NumPy with PPM).

## Limitations / room to grow

- CPU only (swap in CUDA LibTorch for GPU training — the code already auto-selects
  the device).
- Augmentation is horizontal flip + brightness jitter only (mosaic / mixup are not
  implemented — extension points).
- No mAP evaluation (validate visually via detect).
