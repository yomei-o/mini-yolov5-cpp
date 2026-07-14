"""ONNX export bridge for yolov5_cpp.

The C++ trainer writes weights with `yolov5 export-weights` into a portable
binary blob. This script rebuilds an architecturally identical YOLOv5n in
PyTorch (same submodule names -> same state_dict keys), loads those weights,
and runs torch.onnx.export.

Usage:
    python tools/export_onnx.py --weights weights.bin --nc 3 --out model.onnx --imgsz 320
"""
import argparse
import math
import struct

import torch
import torch.nn as nn


# --- modules mirroring src/model/modules.h (names must match!) --------------
def autopad(k, p=None):
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, 1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = torch.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = torch.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = torch.max_pool2d(y2, self.k, 1, self.k // 2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class Detect(nn.Module):
    def __init__(self, nc, anchors, ch):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        a = torch.tensor(anchors).float().view(self.nl, self.na, 2)
        self.register_buffer("anchors", a)  # pixel units; scaled after set_stride
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.stride = torch.zeros(self.nl)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            y = self.m[i](x[i])
            bs, _, ny, nx = y.shape
            y = y.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            y = y.sigmoid()
            yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
            ag = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2)
            xy = (y[..., 0:2] * 2 - 0.5 + grid) * self.stride[i]
            wh = (y[..., 2:4] * 2) ** 2 * ag
            z.append(torch.cat((xy, wh, y[..., 4:]), 4).view(bs, -1, self.no))
        return torch.cat(z, 1)


class YoloV5(nn.Module):
    def __init__(self, nc=80, w=0.25, d=0.33):
        super().__init__()
        cw = lambda b: math.ceil(b * w / 8) * 8
        dn = lambda b: max(1, round(b * d))
        self.conv0 = Conv(3, cw(64), 6, 2, 2)
        self.conv1 = Conv(cw(64), cw(128), 3, 2)
        self.c3_2 = C3(cw(128), cw(128), dn(3))
        self.conv3 = Conv(cw(128), cw(256), 3, 2)
        self.c3_4 = C3(cw(256), cw(256), dn(6))
        self.conv5 = Conv(cw(256), cw(512), 3, 2)
        self.c3_6 = C3(cw(512), cw(512), dn(9))
        self.conv7 = Conv(cw(512), cw(1024), 3, 2)
        self.c3_8 = C3(cw(1024), cw(1024), dn(3))
        self.sppf9 = SPPF(cw(1024), cw(1024), 5)
        self.conv10 = Conv(cw(1024), cw(512), 1, 1)
        self.c3_13 = C3(cw(512) + cw(512), cw(512), dn(3), False)
        self.conv14 = Conv(cw(512), cw(256), 1, 1)
        self.c3_17 = C3(cw(256) + cw(256), cw(256), dn(3), False)
        self.conv18 = Conv(cw(256), cw(256), 3, 2)
        self.c3_20 = C3(cw(256) + cw(256), cw(512), dn(3), False)
        self.conv21 = Conv(cw(512), cw(512), 3, 2)
        self.c3_23 = C3(cw(512) + cw(512), cw(1024), dn(3), False)
        anchors = [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]]
        self.detect = Detect(nc, anchors, [cw(256), cw(512), cw(1024)])
        self._set_stride()

    def _forward_features(self, x):
        x = self.conv0(x); x = self.conv1(x); x = self.c3_2(x); x = self.conv3(x)
        p3 = self.c3_4(x); x = self.conv5(p3); p4 = self.c3_6(x)
        x = self.conv7(p4); x = self.c3_8(x); x = self.sppf9(x)
        h10 = self.conv10(x)
        x = torch.cat((nn.functional.interpolate(h10, scale_factor=2), p4), 1)
        x = self.c3_13(x)
        h14 = self.conv14(x)
        x = torch.cat((nn.functional.interpolate(h14, scale_factor=2), p3), 1)
        o3 = self.c3_17(x)
        x = torch.cat((self.conv18(o3), h14), 1); o4 = self.c3_20(x)
        x = torch.cat((self.conv21(o4), h10), 1); o5 = self.c3_23(x)
        return [o3, o4, o5]

    def forward(self, x):
        return self.detect(self._forward_features(x))

    def _set_stride(self):
        s = 256
        feats = self._forward_features(torch.zeros(1, 3, s, s))
        self.detect.stride = torch.tensor([s / f.shape[-2] for f in feats])
        for i in range(self.detect.nl):
            self.detect.anchors[i] /= self.detect.stride[i]


def load_blob(path):
    """Read the "YW01" blob written by `yolov5 export-weights`."""
    with open(path, "rb") as f:
        assert f.read(4) == b"YW01", "bad magic"
        (count,) = struct.unpack("<i", f.read(4))
        sd = {}
        for _ in range(count):
            (nlen,) = struct.unpack("<i", f.read(4))
            name = f.read(nlen).decode()
            (ndim,) = struct.unpack("<i", f.read(4))
            dims = struct.unpack("<%dq" % ndim, f.read(8 * ndim))
            numel = 1
            for d in dims:
                numel *= d
            data = struct.unpack("<%df" % numel, f.read(4 * numel))
            sd[name] = torch.tensor(data, dtype=torch.float32).view(*dims) if dims \
                else torch.tensor(data[0])
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="weights.bin from export-weights")
    ap.add_argument("--nc", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    model = YoloV5(args.nc, 0.25, 0.33).eval()
    blob = load_blob(args.weights)
    missing, unexpected = model.load_state_dict(blob, strict=False)
    # only num_batches_tracked should be missing; report anything else
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing:
        print("WARNING missing keys:", real_missing)
    if unexpected:
        print("WARNING unexpected keys:", unexpected)
    print(f"loaded {len(blob)} tensors (missing {len(missing)}, unexpected {len(unexpected)})")

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    export_kwargs = dict(
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset,
    )
    try:
        # Use the mature TorchScript exporter (avoids the onnxscript dependency
        # of the newer dynamo path). dynamo kwarg exists on torch >= 2.5.
        torch.onnx.export(model, dummy, args.out, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.out, **export_kwargs)
    print(f"exported ONNX -> {args.out}  (output shape: 1 x {model.detect.no}-vec grid)")


if __name__ == "__main__":
    main()
