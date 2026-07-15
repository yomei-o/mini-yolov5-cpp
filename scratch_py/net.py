"""A compact YOLO-style detector built entirely on the self-made autograd.

Python twin of ``scratch/net.cpp``. Same ideas as YOLOv5 -- BatchNorm, a C3/SPPF
backbone, a 2-scale FPN head (strides 8 and 16), anchors, CIoU box loss -- shrunk
to something a NumPy autograd can actually train. Read alongside ``autograd.py``.
"""

import os
import math
import struct
import zlib
import time
import numpy as np

import autograd as ag
from autograd import Tensor


class Box:
    """A normalized [0,1] box with a class id."""
    __slots__ = ("cls", "cx", "cy", "w", "h")

    def __init__(self, cls, cx, cy, w, h):
        self.cls, self.cx, self.cy, self.w, self.h = cls, cx, cy, w, h


# ---------------------------------------------------------------------------
# image I/O: pure-Python PPM reader, stdlib-zlib PNG writer, optional PIL
# ---------------------------------------------------------------------------
def _load_ppm(path):
    """Read a binary PPM (P6) into planar CHW float [0,1]. Pure Python."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6", "not a P6 PPM"
        # skip comment lines, then read "W H" and maxval
        vals = []
        while len(vals) < 3:
            line = f.readline().split(b"#")[0].split()
            vals += [int(v) for v in line]
        W, H, _ = vals
        buf = np.frombuffer(f.read(W * H * 3), np.uint8).reshape(H, W, 3)
    return buf.transpose(2, 0, 1).astype(np.float64) / 255.0, W, H


def _load_image_any(path):
    """PPM via the pure loader; JPG/PNG/BMP via Pillow if it is installed."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ppm":
        return _load_ppm(path)
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            f"reading {ext} needs Pillow (pip install pillow), or convert to .ppm")
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, np.float64) / 255.0        # HWC
    H, W = arr.shape[:2]
    return arr.transpose(2, 0, 1), W, H


def _png_chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))


def save_png(path, img_chw):
    """Write a (3,S,S) float [0,1] image as a PNG using only stdlib zlib."""
    _, H, W = img_chw.shape
    rgb = (np.clip(img_chw, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    raw = bytearray()
    for y in range(H):
        raw.append(0)                # per-row filter type 0 (None)
        raw.extend(rgb[y].tobytes())
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)))
        f.write(_png_chunk(b"IDAT", zlib.compress(bytes(raw))))
        f.write(_png_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# dataset: letterbox images to SxS and remap boxes (shared YOLO txt labels)
# ---------------------------------------------------------------------------
def _letterbox(src, W, H, S, boxes):
    """Nearest-neighbour letterbox (3,H,W) -> (3,S,S); remap boxes in place."""
    r = min(S / W, S / H)
    nw, nh = round(W * r), round(H * r)
    padx, pady = (S - nw) // 2, (S - nh) // 2
    dst = np.full((3, S, S), 114.0 / 255.0)          # gray pad
    xs = np.minimum(W - 1, (np.arange(nw) / r).astype(int))
    ys = np.minimum(H - 1, (np.arange(nh) / r).astype(int))
    dst[:, pady:pady + nh, padx:padx + nw] = src[:, ys][:, :, xs]
    for b in boxes:
        b.cx = (b.cx * W * r + padx) / S
        b.cy = (b.cy * H * r + pady) / S
        b.w = b.w * W * r / S
        b.h = b.h * H * r / S
    return dst


class Dataset:
    def __init__(self, S):
        self.S = S
        self.imgs = []      # each (3,S,S)
        self.boxes = []     # each: list[Box]

    def __len__(self):
        return len(self.imgs)


def load_disk_dataset(images_dir, S):
    """Load images + YOLO txt labels; accepts either <dir> or <dir>/images."""
    exts = {".ppm", ".jpg", ".jpeg", ".png", ".bmp"}
    idir = images_dir
    if os.path.isdir(os.path.join(images_dir, "images")):
        idir = os.path.join(images_dir, "images")
    ds = Dataset(S)
    for name in sorted(os.listdir(idir)):
        if os.path.splitext(name)[1].lower() not in exts:
            continue
        try:
            rgb, W, H = _load_image_any(os.path.join(idir, name))
        except Exception:
            continue
        # label path: swap the "images" segment for "labels", ext -> .txt
        label = os.path.join(idir.replace("images", "labels"),
                             os.path.splitext(name)[0] + ".txt")
        boxes = []
        if os.path.exists(label):
            with open(label) as lf:
                for line in lf:
                    p = line.split()
                    if len(p) >= 5:
                        boxes.append(Box(int(float(p[0])), float(p[1]), float(p[2]),
                                         float(p[3]), float(p[4])))
        ds.imgs.append(_letterbox(rgb, W, H, S, boxes))
        ds.boxes.append(boxes)
    return ds


def batch_from(ds, N, shuffle):
    """Assemble a minibatch (random if shuffle, else the first N)."""
    imgs = np.zeros((N, 3, ds.S, ds.S), np.float64)
    boxes = []
    for k in range(N):
        idx = int(ag.randf() * len(ds)) % len(ds) if shuffle else k % len(ds)
        imgs[k] = ds.imgs[idx]
        boxes.append(list(ds.boxes[idx]))
    return Tensor(imgs), boxes


# ---------------------------------------------------------------------------
# building blocks: Conv(no bias) -> BatchNorm -> SiLU, plus C3 / SPPF
# ---------------------------------------------------------------------------
class ConvBN:
    """Conv (no bias) -> BatchNorm -> (SiLU applied by `block`). He-style init."""

    def __init__(self, cin, cout, k, stride, pad):
        self.w = Tensor.randn((cout, cin, k, k), math.sqrt(2.0 / (cin * k * k)), True)
        self.gamma = Tensor(np.ones(cout), requires_grad=True)
        self.beta = Tensor.zeros((cout,), requires_grad=True)
        self.run_mean = np.zeros(cout)          # BN buffers (no grad)
        self.run_var = np.ones(cout)
        self.stride, self.pad = stride, pad

    def params(self):
        return [self.w, self.gamma, self.beta]


def block(net, c, x):
    h = ag.conv2d(x, c.w, Tensor.zeros((c.w.shape[0],)), c.stride, c.pad)
    h = ag.batchnorm2d(h, c.gamma, c.beta, c.run_mean, c.run_var, net.training)
    return ag.silu(h)


class Bottleneck:
    def __init__(self, c_):
        self.cv1 = ConvBN(c_, c_, 1, 1, 0)
        self.cv2 = ConvBN(c_, c_, 3, 1, 1)

    def params(self):
        return self.cv1.params() + self.cv2.params()


class C3:
    """CSP block: split -> n bottlenecks on one branch -> concat -> fuse."""

    def __init__(self, cin, cout, n):
        c_ = cout // 2
        self.cv1 = ConvBN(cin, c_, 1, 1, 0)
        self.cv2 = ConvBN(cin, c_, 1, 1, 0)
        self.cv3 = ConvBN(2 * c_, cout, 1, 1, 0)
        self.m = [Bottleneck(c_) for _ in range(n)]

    def params(self):
        p = self.cv1.params() + self.cv2.params() + self.cv3.params()
        for b in self.m:
            p += b.params()
        return p


class SPPF:
    """Spatial pyramid pooling - fast: 3 chained maxpools, then concat + fuse."""

    def __init__(self, cin, cout, k=5):
        self.k = k
        c_ = cin // 2
        self.cv1 = ConvBN(cin, c_, 1, 1, 0)
        self.cv2 = ConvBN(4 * c_, cout, 1, 1, 0)

    def params(self):
        return self.cv1.params() + self.cv2.params()


def fwd_c3(net, m, x):
    y = block(net, m.cv1, x)
    for b in m.m:
        y = ag.add(y, block(net, b.cv2, block(net, b.cv1, y)))   # residual
    y2 = block(net, m.cv2, x)
    return block(net, m.cv3, ag.cat_channels([y, y2]))


def fwd_sppf(net, s, x):
    x1 = block(net, s.cv1, x)
    p1 = ag.maxpool2d(x1, s.k, 1, s.k // 2)
    p2 = ag.maxpool2d(p1, s.k, 1, s.k // 2)
    p3 = ag.maxpool2d(p2, s.k, 1, s.k // 2)
    return block(net, s.cv2, ag.cat_channels([x1, p1, p2, p3]))


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
class SmallYolo:
    def __init__(self, classes=3, input_size=64):
        self.S = input_size
        self.nc = classes
        self.no = 5 + classes
        self.na = 3
        self.training = True
        self.grids = [self.S // 8, self.S // 16]     # stride-8 and stride-16 grids
        self.strides = [8, 16]
        # per-scale anchors in normalized (w,h): [w0,h0,w1,h1,w2,h2]
        self.anchors = [[0.05, 0.08, 0.10, 0.16, 0.18, 0.28],    # stride 8
                        [0.25, 0.30, 0.40, 0.50, 0.65, 0.80]]     # stride 16

        # v5-like backbone: stride-2 conv downsampling + C3 blocks + SPPF
        self.conv0 = ConvBN(3, 16, 3, 2, 1)     # /2
        self.conv1 = ConvBN(16, 32, 3, 2, 1)    # /4
        self.c3a = C3(32, 32, 1)
        self.conv2 = ConvBN(32, 64, 3, 2, 1)    # /8   (stride 8)
        self.c3b = C3(64, 64, 1)                # -> C8
        self.conv3 = ConvBN(64, 128, 3, 2, 1)   # /16  (stride 16)
        self.c3c = C3(128, 128, 1)
        self.sppf = SPPF(128, 128)              # -> C16
        self.c3h = C3(64 + 128, 64, 1)          # FPN head (cat C8 + up(C16))
        # detection heads: 1x1 conv to na*no channels, field-major layout
        self.h8w = Tensor.randn((self.na * self.no, 64, 1, 1), math.sqrt(2.0 / 64), True)
        self.h8b = Tensor.zeros((self.na * self.no,), requires_grad=True)
        self.h16w = Tensor.randn((self.na * self.no, 128, 1, 1), math.sqrt(2.0 / 128), True)
        self.h16b = Tensor.zeros((self.na * self.no,), requires_grad=True)

    def forward(self, x):
        x0 = block(self, self.conv0, x)          # /2
        x1 = block(self, self.conv1, x0)         # /4
        x1 = fwd_c3(self, self.c3a, x1)
        x2 = block(self, self.conv2, x1)         # /8
        c8 = fwd_c3(self, self.c3b, x2)          # stride-8 feature
        x3 = block(self, self.conv3, c8)         # /16
        x3 = fwd_c3(self, self.c3c, x3)
        c16 = fwd_sppf(self, self.sppf, x3)      # stride-16 feature

        up = ag.upsample_nearest2d(c16, 2)       # FPN top-down
        p8 = fwd_c3(self, self.c3h, ag.cat_channels([c8, up]))

        pred8 = ag.conv2d(p8, self.h8w, self.h8b, 1, 0)     # (N, na*no, G8, G8)
        pred16 = ag.conv2d(c16, self.h16w, self.h16b, 1, 0)  # (N, na*no, G16, G16)
        return [pred8, pred16]

    def params(self):
        p = []
        for m in (self.conv0, self.conv1, self.conv2, self.conv3,
                  self.c3a, self.c3b, self.c3c, self.c3h, self.sppf):
            p += m.params()
        p += [self.h8w, self.h8b, self.h16w, self.h16b]
        return p


# 3 classes distinguished by color (matches the LibTorch demo).
_COLOR = np.array([[0.85, 0.15, 0.15],
                   [0.15, 0.80, 0.20],
                   [0.20, 0.30, 0.90]])


def gen_batch(N, S):
    """Synthetic in-memory data: colored rectangles on noisy backgrounds."""
    imgs = np.zeros((N, 3, S, S), np.float64)
    boxes = []
    for n in range(N):
        imgs[n] = 0.35 + 0.12 * (np.random.default_rng(
            int(ag.randf() * 1e9)).random((3, S, S)) - 0.5)
        bs = []
        nobj = 1 + (1 if ag.randf() < 0.5 else 0)
        for _ in range(nobj):
            cls = int(ag.randf() * 3) % 3
            bw = int(S * (0.2 + 0.25 * ag.randf()))
            bh = int(S * (0.2 + 0.25 * ag.randf()))
            x1 = int(ag.randf() * (S - bw))
            y1 = int(ag.randf() * (S - bh))
            for c in range(3):
                imgs[n, c, y1:y1 + bh, x1:x1 + bw] = _COLOR[cls, c]
            bs.append(Box(cls, (x1 + bw / 2) / S, (y1 + bh / 2) / S, bw / S, bh / S))
        boxes.append(bs)
    return Tensor(imgs), boxes


# ---------------------------------------------------------------------------
# differentiable CIoU, built from the gradient-checked primitives
# ---------------------------------------------------------------------------
def bbox_ciou(bx, by, bw, bh, tx, ty, tw, th):
    eps, pi = 1e-7, math.pi

    def half(t):
        return ag.mul_scalar(t, 0.5)

    def sq(t):
        return ag.mul(t, t)

    px1, px2 = ag.sub(bx, half(bw)), ag.add(bx, half(bw))
    py1, py2 = ag.sub(by, half(bh)), ag.add(by, half(bh))
    tx1, tx2 = ag.sub(tx, half(tw)), ag.add(tx, half(tw))
    ty1, ty2 = ag.sub(ty, half(th)), ag.add(ty, half(th))

    iw = ag.clamp_min(ag.sub(ag.minimum(px2, tx2), ag.maximum(px1, tx1)), 0.0)
    ih = ag.clamp_min(ag.sub(ag.minimum(py2, ty2), ag.maximum(py1, ty1)), 0.0)
    inter = ag.mul(iw, ih)
    uni = ag.add_scalar(ag.sub(ag.add(ag.mul(bw, bh), ag.mul(tw, th)), inter), eps)
    iou = ag.divide(inter, uni)

    cw = ag.sub(ag.maximum(px2, tx2), ag.minimum(px1, tx1))
    ch = ag.sub(ag.maximum(py2, ty2), ag.minimum(py1, ty1))
    c2 = ag.add_scalar(ag.add(sq(cw), sq(ch)), eps)
    rho2 = ag.add(sq(ag.sub(bx, tx)), sq(ag.sub(by, ty)))

    at_t = ag.atan_(ag.divide(tw, ag.add_scalar(th, eps)))
    at_p = ag.atan_(ag.divide(bw, ag.add_scalar(bh, eps)))
    v = ag.mul_scalar(sq(ag.sub(at_t, at_p)), 4.0 / (pi * pi))
    denom = ag.add_scalar(ag.add(ag.add_scalar(ag.mul_scalar(iou, -1.0), 1.0), v), eps)
    alpha = ag.divide(v, denom)
    return ag.sub(ag.sub(iou, ag.divide(rho2, c2)), ag.mul(v, alpha))


def _const_grid(N, na, G, mode, anch):
    """Constant (N,na,G,G): mode 0=gx, 1=gy, 2=anchor-w, 3=anchor-h."""
    if mode == 0:
        base = np.broadcast_to(np.arange(G)[None, None, None, :], (N, na, G, G))
    elif mode == 1:
        base = np.broadcast_to(np.arange(G)[None, None, :, None], (N, na, G, G))
    else:
        a = np.array([anch[2 * i + (mode - 2)] for i in range(na)])
        base = np.broadcast_to(a[None, :, None, None], (N, na, G, G))
    return Tensor(np.ascontiguousarray(base, np.float64))


def loss_for_scale(net, pred, s, boxes):
    """Anchor-based YOLO loss for one scale (CIoU box + objectness + class)."""
    N, na, nc, G = pred.shape[0], net.na, net.nc, net.grids[s]
    anch = net.anchors[s]

    # target/mask grids as plain numpy (constants fed to the graph as leaves)
    tcx = np.zeros((N, na, G, G)); tcy = np.zeros((N, na, G, G))
    tw = np.zeros((N, na, G, G)); th = np.zeros((N, na, G, G))
    obj_t = np.zeros((N, na, G, G))
    box_mask = np.zeros((N, na, G, G))
    cls_t = np.zeros((N, nc * na, G, G))
    cls_mask = np.zeros((N, nc * na, G, G))

    def assign(n, a, gy, gx, b):
        tcx[n, a, gy, gx] = b.cx; tcy[n, a, gy, gx] = b.cy
        tw[n, a, gy, gx] = b.w; th[n, a, gy, gx] = b.h
        obj_t[n, a, gy, gx] = 1.0; box_mask[n, a, gy, gx] = 1.0
        for c in range(nc):
            cls_mask[n, c * na + a, gy, gx] = 1.0
        cls_t[n, b.cls * na + a, gy, gx] = 1.0

    npos = 0
    for n in range(N):
        for b in boxes[n]:
            gx = min(G - 1, int(b.cx * G))
            gy = min(G - 1, int(b.cy * G))
            best_a, best_r, any_ = 0, 1e9, False
            for a in range(na):
                rw = max(b.w / anch[2 * a], anch[2 * a] / b.w)
                rh = max(b.h / anch[2 * a + 1], anch[2 * a + 1] / b.h)
                r = max(rw, rh)
                if r < best_r:
                    best_r, best_a = r, a
                if r < 4.0:
                    any_ = True
                    assign(n, a, gy, gx, b)
                    npos += 1
            if not any_:                        # fallback to the best anchor
                assign(n, best_a, gy, gx, b)
                npos += 1

    inv_pos = 1.0 / max(1, npos)
    neg_mask = 1.0 - box_mask
    inv_neg = 1.0 / max(1, N * na * G * G - npos)

    # constant decode grids
    gx_g = _const_grid(N, na, G, 0, anch); gy_g = _const_grid(N, na, G, 1, anch)
    aw_g = _const_grid(N, na, G, 2, anch); ah_g = _const_grid(N, na, G, 3, anch)

    # decode (v5 formula); fields sliced per-anchor-block (field-major layout)
    def slc(f):
        return ag.slice_channels(pred, f * na, (f + 1) * na)

    sx, sy = ag.sigmoid(slc(0)), ag.sigmoid(slc(1))
    sw, sh = ag.sigmoid(slc(2)), ag.sigmoid(slc(3))
    objp = ag.sigmoid(slc(4))
    clsp = ag.sigmoid(ag.slice_channels(pred, 5 * na, net.no * na))   # (N,nc*na,G,G)

    invG = 1.0 / G
    bx = ag.mul_scalar(ag.add(ag.add_scalar(ag.mul_scalar(sx, 2.0), -0.5), gx_g), invG)
    by = ag.mul_scalar(ag.add(ag.add_scalar(ag.mul_scalar(sy, 2.0), -0.5), gy_g), invG)
    bw = ag.mul(ag.mul(ag.mul_scalar(sw, 2.0), ag.mul_scalar(sw, 2.0)), aw_g)  # (2sw)^2 * aw
    bh = ag.mul(ag.mul(ag.mul_scalar(sh, 2.0), ag.mul_scalar(sh, 2.0)), ah_g)

    ciou = bbox_ciou(bx, by, bw, bh,
                     Tensor(tcx), Tensor(tcy), Tensor(tw), Tensor(th))
    box_mask_t = Tensor(box_mask)
    Lbox = ag.mul_scalar(
        ag.sum(ag.mul(box_mask_t, ag.add_scalar(ag.mul_scalar(ciou, -1.0), 1.0))), inv_pos)

    # Objectness target = detached CIoU at positives (YOLOv5-style): confidence
    # reflects box quality, so background / low-IoU cells stay low. ciou.data is
    # the eagerly-computed forward value; obj_t is a constant leaf -> stop-grad.
    pos = box_mask > 0.5
    obj_t[pos] = np.maximum(0.0, ciou.data[pos])

    def sq(d):
        return ag.mul(d, d)

    obj_err = sq(ag.sub(objp, Tensor(obj_t)))
    # negatives weighted 2x to suppress background objectness (fewer FPs)
    Lobj = ag.add(ag.mul_scalar(ag.sum(ag.mul(box_mask_t, obj_err)), inv_pos),
                  ag.mul_scalar(ag.sum(ag.mul(Tensor(neg_mask), obj_err)), inv_neg * 2.0))
    Lcls = ag.mul_scalar(
        ag.sum(ag.mul(Tensor(cls_mask), sq(ag.sub(clsp, Tensor(cls_t))))), inv_pos / nc)

    # box gain 5, class gain 2 (stronger class supervision)
    return ag.add(ag.add(ag.mul_scalar(Lbox, 5.0), Lobj), ag.mul_scalar(Lcls, 2.0))


def _assign_scale(b, nscales):
    """Route each GT to ONE scale by size: small -> fine (8), large -> coarse."""
    sz = max(b.w, b.h)
    return 1 if (nscales > 1 and sz > 0.33) else 0


def compute_loss(net, preds, boxes):
    total = None
    for s in range(len(preds)):
        sub = [[b for b in boxes[n] if _assign_scale(b, len(preds)) == s]
               for n in range(len(boxes))]
        l = loss_for_scale(net, preds[s], s, sub)
        total = l if total is None else ag.add(total, l)
    return total


# ---------------------------------------------------------------------------
# decode + NMS + drawing
# ---------------------------------------------------------------------------
def _sig(v):
    return 1.0 / (1.0 + np.exp(-v))


def decode(net, preds, n, conf_thres):
    """Decode all scales for image n into (boxes, confidences) above threshold."""
    nc, na = net.nc, net.na
    out, confs = [], []
    for s in range(len(preds)):
        G = net.grids[s]
        anch = net.anchors[s]
        d = preds[s].data[n]                    # (na*no, G, G), channel = field*na + a
        for a in range(na):
            for gy in range(G):
                for gx in range(G):
                    obj = _sig(d[4 * na + a, gy, gx])
                    if obj < conf_thres:
                        continue
                    cls_scores = [_sig(d[(5 + c) * na + a, gy, gx]) for c in range(nc)]
                    best = int(np.argmax(cls_scores))
                    sw, sh = _sig(d[2 * na + a, gy, gx]), _sig(d[3 * na + a, gy, gx])
                    bw = (2 * sw) ** 2 * anch[2 * a]
                    bh = (2 * sh) ** 2 * anch[2 * a + 1]
                    if bw * bh < 1e-4:
                        continue
                    cx = (gx + 2 * _sig(d[0 * na + a, gy, gx]) - 0.5) / G
                    cy = (gy + 2 * _sig(d[1 * na + a, gy, gx]) - 0.5) / G
                    out.append(Box(best, cx, cy, bw, bh))
                    confs.append(obj)
    return out, confs


def _iou(p, q):
    ax1, ay1, ax2, ay2 = p.cx - p.w / 2, p.cy - p.h / 2, p.cx + p.w / 2, p.cy + p.h / 2
    bx1, by1, bx2, by2 = q.cx - q.w / 2, q.cy - q.h / 2, q.cx + q.w / 2, q.cy + q.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    uni = p.w * p.h + q.w * q.h - inter
    return inter / uni if uni > 0 else 0.0


def nms(boxes, conf, iou_thr):
    """Greedy per-class NMS; returns indices to keep."""
    order = sorted(range(len(boxes)), key=lambda i: conf[i], reverse=True)
    dead = [False] * len(boxes)
    keep = []
    for idx in order:
        if dead[idx]:
            continue
        keep.append(idx)
        for o in order:
            if (not dead[o] and o != idx and boxes[o].cls == boxes[idx].cls
                    and _iou(boxes[idx], boxes[o]) > iou_thr):
                dead[o] = True
    return keep


def draw_box(img, S, b, color, th=2):
    x1, y1 = int((b.cx - b.w / 2) * S), int((b.cy - b.h / 2) * S)
    x2, y2 = int((b.cx + b.w / 2) * S), int((b.cy + b.h / 2) * S)
    clip = lambda v: max(0, min(S - 1, v))
    x1, x2, y1, y2 = clip(x1), clip(x2), clip(y1), clip(y2)
    col = np.array(color).reshape(3, 1)
    for t in range(th):
        img[:, clip(y1 + t), x1:x2 + 1] = col
        img[:, clip(y2 - t), x1:x2 + 1] = col
        img[:, y1:y2 + 1, clip(x1 + t)] = col
        img[:, y1:y2 + 1, clip(x2 - t)] = col


# ---------------------------------------------------------------------------
# training demo (SGD + momentum + warmup + cosine decay), mirrors main.cpp
# ---------------------------------------------------------------------------
def train_demo(data_dir="", nc=3, input_size=64):
    ag.seed(7)
    net = SmallYolo(nc, input_size)
    params = net.params()
    vel = [np.zeros_like(p.data) for p in params]     # momentum buffers

    lr0, mom = 0.03, 0.9
    N, S, steps, warmup = 12, net.S, 2500, 50

    disk = bool(data_dir)
    ds = None
    if disk:
        ds = load_disk_dataset(data_dir, S)
        if len(ds) == 0:
            print(f"no images under {data_dir}")
            return 1

    print("== train from-scratch YOLO (self-made autograd on NumPy) ==")
    print(f"   input {S}x{S}, grids {net.grids[0]}x{net.grids[0]} + "
          f"{net.grids[1]}x{net.grids[1]} (FPN, BatchNorm), {net.nc} classes, batch {N}")
    print(f"   data: {data_dir + f' (disk, {len(ds)} imgs)' if disk else 'in-memory synthetic'}")

    net.training = True
    t0 = time.time()
    for step in range(steps + 1):
        imgs, boxes = batch_from(ds, N, True) if disk else gen_batch(N, S)
        if step < warmup:                             # linear warmup
            lr = lr0 * step / warmup
        else:                                         # then cosine decay to 5%
            p = (step - warmup) / (steps - warmup)
            lr = lr0 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

        for p in params:
            p.zero_grad()
        preds = net.forward(imgs)
        loss = compute_loss(net, preds, boxes)
        loss.backward()
        for i, p in enumerate(params):                # SGD with momentum
            vel[i] = mom * vel[i] - lr * p.grad
            p.data += vel[i]
        if step % 50 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  lr {lr:.4f}", flush=True)

    secs = time.time() - t0
    print(f"\ntrained {steps} steps in {secs:.1f}s  ({secs / (steps + 1):.3f}s/step)")

    # --- evaluate on fresh images (BN now uses running stats) ---
    print("== evaluate (obj>0.7) ==")
    net.training = False
    neval = min(4, len(ds)) if disk else 4
    imgs, boxes = batch_from(ds, neval, False) if disk else gen_batch(neval, S)
    preds = net.forward(imgs)
    if nc == 3:
        names = ["red", "green", "blue"]
    elif nc == 2:
        names = ["person", "car"]
    else:
        names = [f"cls{c}" for c in range(nc)]
    nm = lambda c: names[c] if 0 <= c < nc else "?"

    hits = total = 0
    for n in range(neval):
        print(f" image {n}:")
        canvas = imgs.data[n].copy()
        for g in boxes[n]:
            print(f"   GT  {nm(g.cls):<6} cx={g.cx:.2f} cy={g.cy:.2f} w={g.w:.2f} h={g.h:.2f}")
            draw_box(canvas, S, g, (1, 0, 0), 1)      # GT = thin red
            total += 1
        dets, confs = decode(net, preds, n, 0.7)
        for k in nms(dets, confs, 0.3):
            db = dets[k]
            print(f"   DET {nm(db.cls):<6} cx={db.cx:.2f} cy={db.cy:.2f} "
                  f"w={db.w:.2f} h={db.h:.2f}  conf={confs[k]:.2f}")
            draw_box(canvas, S, db, (0, 1, 0), 2)     # DET = thick green
            for g in boxes[n]:
                if g.cls == db.cls and abs(g.cx - db.cx) < 0.15 and abs(g.cy - db.cy) < 0.15:
                    hits += 1
                    break
        vp = f"viz_{n}.png"
        save_png(vp, canvas)
        print(f"   -> wrote {vp} (GT=red, DET=green)")
    print(f"\n matched {hits} detections to GT (of {total} GT objects)")
    return 0
