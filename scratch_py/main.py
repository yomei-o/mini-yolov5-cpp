"""Entry point: gradient checks (default) or the training demo (`train`).

    python main.py                 # numeric gradient check of every op
    python main.py train           # train on in-memory synthetic data (nc=3, 64px)
    python main.py train DIR       # train on a shared YOLO dataset at DIR (nc=3, 64px)
    python main.py train DIR NC S  # ...with NC classes at S x S input

Mirrors ``scratch/main.cpp``. The gradient check is how we *prove* every
hand-written backward is correct before trusting it to train anything.
"""

import sys
import numpy as np
import autograd as ag
from autograd import Tensor


def gradcheck(inputs, build_loss, eps=1e-3):
    """Compare analytic backward() grads against central finite differences."""
    for t in inputs:
        t.zero_grad()
    loss = build_loss()
    loss.backward()
    analytic = [t.grad.copy() for t in inputs]

    max_err = 0.0
    for ti, t in enumerate(inputs):
        flat = t.data.reshape(-1)
        aflat = analytic[ti].reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            fp = build_loss().item()
            flat[i] = orig - eps
            fm = build_loss().item()
            flat[i] = orig
            num = (fp - fm) / (2 * eps)
            a = aflat[i]
            # relative error for O(1)+ grads, absolute for tiny ones
            denom = max(1.0, abs(a) + abs(num))
            max_err = max(max_err, abs(a - num) / denom)
    return max_err


def check(name, err):
    ok = "OK" if err < 1e-2 else "*** FAIL ***"
    print(f"  {name:<22} max rel err = {err:.2e}   {ok}")


def run_gradchecks():
    ag.seed(42)
    print("== gradient check (analytic backward vs numeric) ==")

    # add/mul/sub composition: sum((a+b)*(a-b)) = sum(a^2 - b^2)
    a = Tensor.randn((3, 4), 1, True)
    b = Tensor.randn((3, 4), 1, True)
    check("add*mul*sub", gradcheck([a, b],
          lambda: ag.sum(ag.mul(ag.add(a, b), ag.sub(a, b)))))

    # matmul
    a = Tensor.randn((4, 5), 1, True)
    b = Tensor.randn((5, 3), 1, True)
    check("matmul", gradcheck([a, b], lambda: ag.sum(ag.matmul(a, b))))

    # activations
    a = Tensor.randn((2, 6), 1, True)
    check("silu", gradcheck([a], lambda: ag.sum(ag.silu(a))))
    check("sigmoid", gradcheck([a], lambda: ag.sum(ag.sigmoid(a))))
    check("relu", gradcheck([a], lambda: ag.sum(ag.relu(a)), 1e-2))

    # conv2d (the hardest backward): input, weight, bias all requires_grad
    x = Tensor.randn((1, 2, 5, 5), 1, True)
    w = Tensor.randn((3, 2, 3, 3), 0.5, True)
    b = Tensor.randn((3,), 0.5, True)
    check("conv2d", gradcheck([x, w, b],
          lambda: ag.sum(ag.silu(ag.conv2d(x, w, b, 1, 1)))))

    # maxpool + upsample + channel concat/slice
    x = Tensor.randn((1, 2, 6, 6), 1, True)
    check("maxpool2d", gradcheck([x], lambda: ag.sum(ag.maxpool2d(x, 2, 2, 0))))
    check("upsample", gradcheck([x], lambda: ag.sum(ag.upsample_nearest2d(x, 2))))
    y = Tensor.randn((1, 3, 6, 6), 1, True)
    check("cat_channels", gradcheck([x, y],
          lambda: ag.sum(ag.mul(ag.cat_channels([x, y]), ag.cat_channels([x, y])))))
    check("slice_channels", gradcheck([y],
          lambda: ag.sum(ag.slice_channels(y, 1, 3))))

    # batchnorm (training mode: output depends on batch statistics)
    x = Tensor.randn((2, 3, 4, 4), 1, True)
    g = Tensor.randn((3,), 0.5, True)
    b = Tensor.randn((3,), 0.5, True)
    rm = np.zeros(3, np.float32)
    rv = np.ones(3, np.float32)
    check("batchnorm2d", gradcheck([x, g, b],
          lambda: ag.sum(ag.silu(ag.batchnorm2d(x, g, b, rm, rv, True)))))

    # CIoU building-block ops
    a = Tensor.randn((2, 5), 1, True)
    b = Tensor.randn((2, 5), 1, True)
    check("maximum", gradcheck([a, b], lambda: ag.sum(ag.maximum(a, b))))
    check("minimum", gradcheck([a, b], lambda: ag.sum(ag.minimum(a, b))))
    check("atan", gradcheck([a], lambda: ag.sum(ag.atan_(a))))
    pa = Tensor.from_data([1.2, 0.7, 2.1, 0.4], True)
    pb = Tensor.from_data([0.9, 1.5, 0.6, 1.1], True)
    check("divide", gradcheck([pa, pb], lambda: ag.sum(ag.divide(pa, pb))))
    check("sqrt", gradcheck([pa], lambda: ag.sum(ag.sqrt_(pa))))
    check("clamp_min", gradcheck([a], lambda: ag.sum(ag.clamp_min(a, 0.0)), 1e-2))

    # a 2-layer MLP with SiLU -> scalar MSE (exercises the whole chain)
    x = Tensor.randn((4, 3), 1, False)
    W1 = Tensor.randn((3, 8), 0.5, True)
    W2 = Tensor.randn((8, 1), 0.5, True)
    y = Tensor.randn((4, 1), 1, False)

    def mlp_loss():
        h = ag.silu(ag.matmul(x, W1))
        pred = ag.matmul(h, W2)
        diff = ag.sub(pred, y)
        return ag.mean(ag.mul(diff, diff))
    check("mlp mse", gradcheck([W1, W2], mlp_loss))

    # tiny end-to-end training: fit y = X w* with SGD on our own autograd
    print("\n== train a linear model with the self-made autograd ==")
    ag.seed(1)
    N, D = 64, 3
    X = Tensor.randn((N, D), 1.0, False)
    wstar = np.array([2.0, -3.0, 0.5], np.float32)
    Y = Tensor(X.data @ wstar.reshape(D, 1), False)
    W = Tensor.randn((D, 1), 0.1, True)
    lr = 0.05
    for step in range(201):
        W.zero_grad()
        pred = ag.matmul(X, W)
        diff = ag.sub(pred, Y)
        loss = ag.mean(ag.mul(diff, diff))
        loss.backward()
        W.data -= lr * W.grad
        if step % 50 == 0:
            w = W.data.reshape(-1)
            print(f"  step {step:3d}  loss {loss.item():.5f}  "
                  f"W=[{w[0]:.3f} {w[1]:.3f} {w[2]:.3f}]")
    print("  target W* = [2.000 -3.000 0.500]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        import net
        data_dir = sys.argv[2] if len(sys.argv) > 2 else ""
        nc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        S = int(sys.argv[4]) if len(sys.argv) > 4 else 64
        sys.exit(net.train_demo(data_dir, nc, S))
    run_gradchecks()
