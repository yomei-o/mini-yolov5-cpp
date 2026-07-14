"""Convert an official Ultralytics YOLOv5 .pt checkpoint into the portable
weights.bin that yolov5_cpp `import-weights` consumes.

Why this is needed: the official checkpoint pickles a DetectionModel whose
layers are named model.0 .. model.24, while yolov5_cpp names them conv0, c3_2,
... , detect. The architectures are identical, so we just remap the top-level
index to our name, load into the mirror PyTorch model, and dump weights.bin.

No yolov5 repo required: we fabricate stub `models.*`/`utils.*` modules so the
checkpoint unpickles far enough to read its tensors.

Usage:
    python tools/convert_ultralytics.py --pt yolov5n.pt --out weights.bin
    # then in C++:
    #   yolov5 import-weights --in weights.bin --nc 80 --out yolov5n_cpp.pt
    #   yolov5 detect --weights yolov5n_cpp.pt --nc 80 --source img.jpg
"""
import argparse
import importlib.abc
import importlib.machinery
import struct
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, __import__("os").path.dirname(__file__))
import export_onnx as E  # our mirror YOLOv5 (same submodule names)


# --- fabricate stub modules so the official checkpoint unpickles ------------
class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (nn.Module,), {})
        cls.__module__ = self.__name__
        setattr(self, name, cls)
        return cls


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIXES = ("models", "utils")

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in self.PREFIXES:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


# model.<idx> -> our submodule name. Indices 11,12,15,16,19,22 are
# Upsample/Concat (no parameters) and are simply absent here.
IDX2NAME = {
    0: "conv0", 1: "conv1", 2: "c3_2", 3: "conv3", 4: "c3_4", 5: "conv5",
    6: "c3_6", 7: "conv7", 8: "c3_8", 9: "sppf9", 10: "conv10", 13: "c3_13",
    14: "conv14", 17: "c3_17", 18: "conv18", 20: "c3_20", 21: "conv21",
    23: "c3_23", 24: "detect",
}


def save_blob(model, path):
    """Write float tensors of model.state_dict() in the C++ 'YW01' format."""
    sd = model.state_dict()
    items = [(k, v) for k, v in sd.items() if v.is_floating_point()]
    with open(path, "wb") as f:
        f.write(b"YW01")
        f.write(struct.pack("<i", len(items)))
        for k, v in items:
            v = v.detach().cpu().contiguous().float()
            f.write(struct.pack("<i", len(k)))
            f.write(k.encode())
            f.write(struct.pack("<i", v.dim()))
            for d in v.shape:
                f.write(struct.pack("<q", d))
            f.write(v.numpy().tobytes())
    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="official yolov5n.pt")
    ap.add_argument("--out", default="weights.bin")
    ap.add_argument("--use-ema", action="store_true", default=True,
                    help="prefer the EMA weights (what Ultralytics uses at inference)")
    args = ap.parse_args()

    sys.meta_path.insert(0, _StubFinder())
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    src = (ck.get("ema") if args.use_ema and ck.get("ema") is not None
           else ck.get("model"))
    src = src.float()
    off = src.state_dict()

    # infer nc from the detection head: m.0.weight is (na*(nc+5), C, 1, 1)
    head = off["model.24.m.0.weight"]
    nc = head.shape[0] // 3 - 5
    print(f"official checkpoint: {len(off)} tensors, inferred nc={nc}")

    # remap model.<idx>.<rest> -> <name>.<rest>
    remapped = {}
    for k, v in off.items():
        parts = k.split(".")            # ["model", idx, ...rest]
        if parts[0] != "model":
            continue
        idx = int(parts[1])
        if idx not in IDX2NAME:
            continue                    # anchor_grid on Detect etc. handled below
        remapped[IDX2NAME[idx] + "." + ".".join(parts[2:])] = v

    model = E.YoloV5(nc, 0.25, 0.33)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    real_unexpected = [k for k in unexpected if "anchor_grid" not in k]
    print(f"load into mirror model: missing {len(missing)} "
          f"(non-trivial {len(real_missing)}), unexpected {len(unexpected)} "
          f"(non-trivial {len(real_unexpected)})")
    if real_missing:
        print("  MISSING:", real_missing[:10])
    if real_unexpected:
        print("  UNEXPECTED:", real_unexpected[:10])

    n = save_blob(model, args.out)
    print(f"wrote {n} tensors -> {args.out}  (nc={nc})")


if __name__ == "__main__":
    main()
