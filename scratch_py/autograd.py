"""A tiny reverse-mode autograd engine on top of NumPy (no deep-learning deps).

This is the Python twin of ``scratch/autograd.cpp``. A ``Tensor`` remembers the
NumPy array it holds *and* how it was computed, so calling ``backward()`` on a
scalar populates ``.grad`` on every tensor that fed into it. NumPy only does the
raw array arithmetic; the calculus (the ``backward_fn`` of every op) is written
out by hand here -- that is the whole point of the "understand training by
building it" from-scratch YOLO.

Design mirrors the C++ version op-for-op so the two can be read side by side:
same ops, same shapes (NCHW), same conv-via-im2col, same BatchNorm backward.
The only real difference is that dense loops become vectorised NumPy.
"""

import numpy as np

# NumPy's natural precision. Unlike the C++ port (float32, for SIMD/cache),
# here float64 keeps the gradient checks crisp -- finite differences don't drown
# in float rounding -- which is the whole pedagogical point of this file.
DTYPE = np.float64

# ---- deterministic RNG so runs are reproducible ---------------------------
# (The C++ version uses xorshift; here we just use a seeded NumPy Generator.
#  Numbers won't match the C++ run, but every Python run with the same seed
#  is identical.)
_rng = np.random.default_rng(0)


def seed(s):
    global _rng
    _rng = np.random.default_rng(s)


def randf():
    """Uniform float in [0, 1)."""
    return float(_rng.random())


# ---------------------------------------------------------------------------
# Tensor: value + gradient + the closure that pushes gradient to its parents
# ---------------------------------------------------------------------------
class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), op="leaf"):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.parents = tuple(parents)      # keep the graph alive / reachable
        self.backward_fn = None            # accumulates into parents' .grad
        self.op = op

    # ---- factories --------------------------------------------------------
    @staticmethod
    def zeros(shape, requires_grad=False):
        return Tensor(np.zeros(shape, DTYPE), requires_grad)

    @staticmethod
    def from_data(data, requires_grad=False):
        return Tensor(np.array(data, DTYPE), requires_grad)

    @staticmethod
    def randn(shape, std=1.0, requires_grad=False):
        return Tensor(_rng.standard_normal(shape).astype(DTYPE) * std, requires_grad)

    @property
    def shape(self):
        return self.data.shape

    def item(self):
        return float(self.data.reshape(-1)[0])

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ---- reverse-mode backward -------------------------------------------
    def backward(self):
        # Build reverse-topological order over the graph rooted at self.
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v.parents:
                build(p)
            topo.append(v)

        build(self)

        # Seed dLoss/dLoss = 1 on the (scalar) root, then walk the graph
        # backwards calling each op's hand-written gradient rule.
        self.grad = np.zeros_like(self.data)
        self.grad.reshape(-1)[0] = 1.0
        for v in reversed(topo):
            if v.backward_fn is not None:
                v.backward_fn()

    # operator sugar (same-shape elementwise, like the C++ operators)
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)


def _op(data, parents, op):
    """Build a fresh output node wired to `parents`; caller sets backward_fn."""
    return Tensor(data, parents=parents, op=op)


# ---------------------------------------------------------------------------
# elementwise & reductions
# ---------------------------------------------------------------------------
def add(a, b):
    out = _op(a.data + b.data, (a, b), "add")

    def bwd():
        a.grad += out.grad
        b.grad += out.grad
    out.backward_fn = bwd
    return out


def sub(a, b):
    out = _op(a.data - b.data, (a, b), "sub")

    def bwd():
        a.grad += out.grad
        b.grad -= out.grad
    out.backward_fn = bwd
    return out


def mul(a, b):
    out = _op(a.data * b.data, (a, b), "mul")

    def bwd():
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    out.backward_fn = bwd
    return out


def mul_scalar(a, s):
    out = _op(a.data * s, (a,), "mul_scalar")

    def bwd():
        a.grad += s * out.grad
    out.backward_fn = bwd
    return out


def add_scalar(a, s):
    out = _op(a.data + s, (a,), "add_scalar")

    def bwd():
        a.grad += out.grad
    out.backward_fn = bwd
    return out


def sum(a):
    out = _op(np.array([a.data.sum()], DTYPE), (a,), "sum")

    def bwd():
        a.grad += out.grad[0]     # broadcast the single scalar grad to all
    out.backward_fn = bwd
    return out


def mean(a):
    return mul_scalar(sum(a), 1.0 / a.data.size)


# ---------------------------------------------------------------------------
# matmul  (m,k) x (k,n) -> (m,n)
# ---------------------------------------------------------------------------
def matmul(a, b):
    out = _op(a.data @ b.data, (a, b), "matmul")

    def bwd():
        a.grad += out.grad @ b.data.T        # dA = dO @ B^T
        b.grad += a.data.T @ out.grad        # dB = A^T @ dO
    out.backward_fn = bwd
    return out


# ---------------------------------------------------------------------------
# conv / pooling (NCHW), conv done as im2col + GEMM (matmul), like the C++ code
# ---------------------------------------------------------------------------
def _im2col(x, kh, kw, stride, pad, OH, OW):
    """Unfold receptive fields: (N,C,H,W) -> (N, C*kh*kw, OH*OW).

    The K axis is ordered (c, i, j) exactly like the C++ ((c*kh+i)*kw+j), so a
    weight reshaped to (O, C*kh*kw) lines up with it.
    """
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    col = np.empty((N, C, kh, kw, OH, OW), DTYPE)
    for i in range(kh):
        for j in range(kw):
            col[:, :, i, j, :, :] = xp[:, :, i:i + stride * OH:stride,
                                          j:j + stride * OW:stride]
    return col.reshape(N, C * kh * kw, OH * OW)


def _col2im(dcol, x_shape, kh, kw, stride, pad, OH, OW):
    """Transpose of im2col: scatter (N, C*kh*kw, OH*OW) back into (N,C,H,W)."""
    N, C, H, W = x_shape
    dcol = dcol.reshape(N, C, kh, kw, OH, OW)
    dxp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), DTYPE)
    for i in range(kh):
        for j in range(kw):
            dxp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride] += \
                dcol[:, :, i, j, :, :]
    return dxp[:, :, pad:pad + H, pad:pad + W]


def conv2d(x, w, b, stride, pad):
    """x:(N,C,H,W) w:(O,C,kh,kw) b:(O,) -> (N,O,OH,OW)."""
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1

    col = _im2col(x.data, kh, kw, stride, pad, OH, OW)   # (N, K, P)
    wm = w.data.reshape(O, -1)                           # (O, K)
    y = np.einsum("ok,nkp->nop", wm, col)                # (N, O, P)
    y += b.data[None, :, None]
    out = _op(y.reshape(N, O, OH, OW), (x, w, b), "conv2d")

    def bwd():
        g = out.grad.reshape(N, O, OH * OW)              # dOut (N,O,P)
        b.grad += g.sum(axis=(0, 2))                     # dBias
        # dW(O,K) = sum_n dOut(N,O,P) . col(N,K,P)^T
        w.grad += np.einsum("nop,nkp->ok", g, col).reshape(w.shape)
        # dcol(N,K,P) = W(O,K)^T . dOut(N,O,P) ; then col2im back to the image
        dcol = np.einsum("ok,nop->nkp", wm, g)
        x.grad += _col2im(dcol, x.shape, kh, kw, stride, pad, OH, OW)
    out.backward_fn = bwd
    return out


def maxpool2d(x, k, stride, pad):
    N, C, H, W = x.shape
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    xp = np.pad(x.data, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                constant_values=-1e30)
    # gather the k*k window for every output cell, then take the max
    win = np.empty((N, C, k, k, OH, OW), DTYPE)
    for i in range(k):
        for j in range(k):
            win[:, :, i, j, :, :] = xp[:, :, i:i + stride * OH:stride,
                                          j:j + stride * OW:stride]
    win = win.reshape(N, C, k * k, OH, OW)
    argmax = win.argmax(axis=2)                          # route grads in bwd
    out = _op(win.max(axis=2), (x,), "maxpool2d")

    def bwd():
        # scatter each output grad back to the argmax position inside its window
        dwin = np.zeros((N, C, k * k, OH, OW), DTYPE)
        idx = np.indices((N, C, OH, OW))
        dwin[idx[0], idx[1], argmax, idx[2], idx[3]] = out.grad
        dwin = dwin.reshape(N, C, k, k, OH, OW)
        dxp = np.zeros_like(xp)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride] += \
                    dwin[:, :, i, j, :, :]
        x.grad += dxp[:, :, pad:pad + H, pad:pad + W]
    out.backward_fn = bwd
    return out


def upsample_nearest2d(x, scale):
    N, C, H, W = x.shape
    y = np.repeat(np.repeat(x.data, scale, axis=2), scale, axis=3)
    out = _op(y, (x,), "upsample")

    def bwd():
        # each source pixel receives the sum of its scale*scale copies
        g = out.grad.reshape(N, C, H, scale, W, scale)
        x.grad += g.sum(axis=(3, 5))
    out.backward_fn = bwd
    return out


def batchnorm2d(x, gamma, beta, running_mean, running_var,
                training, momentum=0.1, eps=1e-5):
    """Per-channel BatchNorm over (N,C,H,W).

    gamma/beta are learnable (C,); running_mean/running_var are plain NumPy
    buffers (no grad) updated in place while training.
    """
    N, C, H, W = x.shape
    M = N * H * W
    xd = x.data
    if training:
        mu = xd.mean(axis=(0, 2, 3))                     # (C,)
        var = xd.var(axis=(0, 2, 3))
        running_mean[:] = (1 - momentum) * running_mean + momentum * mu
        running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mu = running_mean
        var = running_var
    inv = 1.0 / np.sqrt(var + eps)                       # (C,)
    mu_ = mu[None, :, None, None]
    inv_ = inv[None, :, None, None]
    g_ = gamma.data[None, :, None, None]
    xhat = (xd - mu_) * inv_
    out = _op(g_ * xhat + beta.data[None, :, None, None], (x, gamma, beta), "batchnorm")

    def bwd():
        dy = out.grad
        beta.grad += dy.sum(axis=(0, 2, 3))
        gamma.grad += (dy * xhat).sum(axis=(0, 2, 3))
        sum_dy = dy.sum(axis=(0, 2, 3))[None, :, None, None]
        sum_dy_xhat = (dy * xhat).sum(axis=(0, 2, 3))[None, :, None, None]
        # dx = gamma*inv/M * (M*dy - sum(dy) - xhat*sum(dy*xhat))
        x.grad += (g_ * inv_ / M) * (M * dy - sum_dy - xhat * sum_dy_xhat)
    out.backward_fn = bwd
    return out


def cat_channels(xs):
    """Concatenate a list of (N,C_i,H,W) along channels."""
    out = _op(np.concatenate([t.data for t in xs], axis=1),
              tuple(xs), "cat")
    sizes = [t.shape[1] for t in xs]

    def bwd():
        off = 0
        for t, c in zip(xs, sizes):
            t.grad += out.grad[:, off:off + c]
            off += c
    out.backward_fn = bwd
    return out


def slice_channels(x, c0, c1):
    """Channels [c0, c1) of (N,C,H,W)."""
    out = _op(x.data[:, c0:c1], (x,), "slice")

    def bwd():
        x.grad[:, c0:c1] += out.grad
    out.backward_fn = bwd
    return out


# ---------------------------------------------------------------------------
# elementwise math used to build a differentiable CIoU from checked pieces
# ---------------------------------------------------------------------------
def maximum(a, b):
    out = _op(np.maximum(a.data, b.data), (a, b), "maximum")

    def bwd():
        mask = a.data >= b.data
        a.grad += np.where(mask, out.grad, 0)
        b.grad += np.where(mask, 0, out.grad)
    out.backward_fn = bwd
    return out


def minimum(a, b):
    out = _op(np.minimum(a.data, b.data), (a, b), "minimum")

    def bwd():
        mask = a.data <= b.data
        a.grad += np.where(mask, out.grad, 0)
        b.grad += np.where(mask, 0, out.grad)
    out.backward_fn = bwd
    return out


def divide(a, b):
    out = _op(a.data / b.data, (a, b), "divide")

    def bwd():
        a.grad += out.grad / b.data
        b.grad += -out.grad * a.data / (b.data * b.data)
    out.backward_fn = bwd
    return out


def sqrt_(a):
    out = _op(np.sqrt(a.data), (a,), "sqrt")

    def bwd():
        a.grad += out.grad * 0.5 / (out.data + 1e-12)
    out.backward_fn = bwd
    return out


def atan_(a):
    out = _op(np.arctan(a.data), (a,), "atan")

    def bwd():
        a.grad += out.grad / (1 + a.data * a.data)
    out.backward_fn = bwd
    return out


def clamp_min(a, m):
    out = _op(np.maximum(a.data, m), (a,), "clamp_min")

    def bwd():
        a.grad += np.where(a.data > m, out.grad, 0)
    out.backward_fn = bwd
    return out


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def relu(a):
    out = _op(np.maximum(a.data, 0), (a,), "relu")

    def bwd():
        a.grad += (out.data > 0) * out.grad
    out.backward_fn = bwd
    return out


def sigmoid(a):
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = _op(s, (a,), "sigmoid")

    def bwd():
        a.grad += out.data * (1 - out.data) * out.grad
    out.backward_fn = bwd
    return out


def silu(a):
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = _op(a.data * s, (a,), "silu")

    def bwd():
        # d/dx (x*sigmoid(x)) = sigmoid + x*sigmoid*(1-sigmoid)
        a.grad += (s + a.data * s * (1 - s)) * out.grad
    out.backward_fn = bwd
    return out
