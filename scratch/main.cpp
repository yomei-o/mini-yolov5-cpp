#include "autograd.h"
#include "net.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace ag;

// Numerical gradient check: compare backward()'s analytic grads against
// central finite differences. This is how we *prove* the engine is correct.
static float gradcheck(const std::vector<Tensor>& inputs,
                       const std::function<Tensor()>& build_loss,
                       float eps = 1e-3f) {
    // analytic
    for (auto& t : inputs) t.zero_grad();
    Tensor loss = build_loss();
    loss.backward();
    std::vector<std::vector<float>> analytic;
    for (auto& t : inputs) analytic.push_back(t.grad());

    float max_err = 0.f;
    for (size_t ti = 0; ti < inputs.size(); ++ti) {
        auto& d = inputs[ti].data();
        for (size_t i = 0; i < d.size(); ++i) {
            float orig = d[i];
            d[i] = orig + eps;
            float fp = build_loss().item();
            d[i] = orig - eps;
            float fm = build_loss().item();
            d[i] = orig;
            float num = (fp - fm) / (2 * eps);
            float a = analytic[ti][i];
            // normalized error: relative for O(1)+ grads, absolute for tiny ones
            // (near-zero grads in float32 are dominated by finite-diff noise).
            float denom = std::max(1.0f, std::fabs(a) + std::fabs(num));
            max_err = std::max(max_err, std::fabs(a - num) / denom);
        }
    }
    return max_err;
}

static void check(const char* name, float err) {
    printf("  %-22s max rel err = %.2e   %s\n", name, err,
           err < 1e-2f ? "OK" : "*** FAIL ***");
}

int main(int argc, char** argv) {
    // `train [DIR [nc [S]]]` runs the detector (DIR = shared PPM dataset, else
    // synthetic); default (no args) runs the gradient checks.
    if (argc > 1 && std::strcmp(argv[1], "train") == 0) {
        std::string dir = argc > 2 ? argv[2] : "";
        int nc = argc > 3 ? std::atoi(argv[3]) : 3;
        int S = argc > 4 ? std::atoi(argv[4]) : 64;
        return scratch::train_demo(dir, nc, S);
    }

    seed(42);
    printf("== gradient check (analytic backward vs numeric) ==\n");

    {  // add/mul/sub composition
        auto a = Tensor::randn({3, 4}, 1, true);
        auto b = Tensor::randn({3, 4}, 1, true);
        check("add*mul*sub", gradcheck({a, b}, [&] {
            return sum(mul(add(a, b), sub(a, b)));  // sum((a+b)*(a-b)) = sum(a^2-b^2)
        }));
    }
    {  // matmul
        auto a = Tensor::randn({4, 5}, 1, true);
        auto b = Tensor::randn({5, 3}, 1, true);
        check("matmul", gradcheck({a, b}, [&] { return sum(matmul(a, b)); }));
    }
    {  // activations
        auto a = Tensor::randn({2, 6}, 1, true);
        check("silu", gradcheck({a}, [&] { return sum(silu(a)); }));
        check("sigmoid", gradcheck({a}, [&] { return sum(sigmoid(a)); }));
        check("relu", gradcheck({a}, [&] { return sum(relu(a)); }, 1e-2f));
    }
    {  // conv2d (the hardest backward): input, weight, bias all requires_grad
        auto x = Tensor::randn({1, 2, 5, 5}, 1, true);
        auto w = Tensor::randn({3, 2, 3, 3}, 0.5, true);
        auto b = Tensor::randn({3}, 0.5, true);
        check("conv2d", gradcheck({x, w, b}, [&] {
            return sum(silu(conv2d(x, w, b, 1, 1)));
        }));
    }
    {  // maxpool + upsample + channel concat/slice
        auto x = Tensor::randn({1, 2, 6, 6}, 1, true);
        check("maxpool2d", gradcheck({x}, [&] { return sum(maxpool2d(x, 2, 2, 0)); }));
        check("upsample", gradcheck({x}, [&] { return sum(upsample_nearest2d(x, 2)); }));
        auto y = Tensor::randn({1, 3, 6, 6}, 1, true);
        check("cat_channels", gradcheck({x, y}, [&] {
            return sum(mul(cat_channels({x, y}), cat_channels({x, y})));
        }));
        check("slice_channels", gradcheck({y}, [&] {
            return sum(slice_channels(y, 1, 3));
        }));
    }
    {  // batchnorm (training mode: output depends on batch statistics)
        auto x = Tensor::randn({2, 3, 4, 4}, 1, true);
        auto g = Tensor::randn({3}, 0.5, true);
        auto b = Tensor::randn({3}, 0.5, true);
        auto rm = Tensor::zeros({3}), rv = Tensor::zeros({3});
        for (int i = 0; i < 3; ++i) rv.data()[i] = 1.f;
        check("batchnorm2d", gradcheck({x, g, b}, [&] {
            return sum(silu(batchnorm2d(x, g, b, rm, rv, /*training=*/true)));
        }));
    }
    {  // CIoU building-block ops
        auto a = Tensor::randn({2, 5}, 1, true);
        auto b = Tensor::randn({2, 5}, 1, true);
        check("maximum", gradcheck({a, b}, [&] { return sum(maximum(a, b)); }));
        check("minimum", gradcheck({a, b}, [&] { return sum(minimum(a, b)); }));
        check("atan", gradcheck({a}, [&] { return sum(atan_(a)); }));
        // positive inputs for div / sqrt
        auto pa = Tensor::from({1.2f, 0.7f, 2.1f, 0.4f}, {4}, true);
        auto pb = Tensor::from({0.9f, 1.5f, 0.6f, 1.1f}, {4}, true);
        check("divide", gradcheck({pa, pb}, [&] { return sum(divide(pa, pb)); }));
        check("sqrt", gradcheck({pa}, [&] { return sum(sqrt_(pa)); }));
        check("clamp_min", gradcheck({a}, [&] { return sum(clamp_min(a, 0.f)); }, 1e-2f));
    }
    {  // a 2-layer MLP with SiLU -> scalar loss (exercises the whole chain)
        auto x = Tensor::randn({4, 3}, 1, false);
        auto W1 = Tensor::randn({3, 8}, 0.5, true);
        auto W2 = Tensor::randn({8, 1}, 0.5, true);
        auto y = Tensor::randn({4, 1}, 1, false);
        check("mlp mse", gradcheck({W1, W2}, [&] {
            auto h = silu(matmul(x, W1));
            auto pred = matmul(h, W2);
            auto diff = sub(pred, y);
            return mean(mul(diff, diff));
        }));
    }

    // --- tiny end-to-end training: fit y = X w* with SGD on our own autograd ---
    printf("\n== train a linear model with the self-made autograd ==\n");
    seed(1);
    int N = 64, D = 3;
    auto X = Tensor::randn({N, D}, 1.0f, false);
    std::vector<float> wstar = {2.0f, -3.0f, 0.5f};
    auto Y = Tensor::zeros({N, 1}, false);
    for (int i = 0; i < N; ++i) {
        float t = 0;
        for (int j = 0; j < D; ++j) t += X.data()[i * D + j] * wstar[j];
        Y.data()[i] = t;
    }
    auto W = Tensor::randn({D, 1}, 0.1f, true);
    float lr = 0.05f;
    for (int step = 0; step <= 200; ++step) {
        W.zero_grad();
        auto pred = matmul(X, W);
        auto diff = sub(pred, Y);
        auto loss = mean(mul(diff, diff));
        loss.backward();
        for (int j = 0; j < D; ++j) W.data()[j] -= lr * W.grad()[j];
        if (step % 50 == 0)
            printf("  step %3d  loss %.5f  W=[%.3f %.3f %.3f]\n", step, loss.item(),
                   W.data()[0], W.data()[1], W.data()[2]);
    }
    printf("  target W* = [2.000 -3.000 0.500]\n");
    return 0;
}
