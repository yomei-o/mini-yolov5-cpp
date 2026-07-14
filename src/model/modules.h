#pragma once
#include <torch/torch.h>
#include <cmath>

// Core building blocks of YOLOv5, ported 1:1 from the PyTorch models/common.py.
// Each is a torch::nn::Module so autograd / parameter registration work exactly
// like the Python version.
namespace yolo {

// Same-padding for odd kernels (== k//2), matching Python autopad().
inline int autopad(int k, int p = -1) { return p >= 0 ? p : k / 2; }

// Round channel count up to a multiple of `divisor` (YOLOv5 make_divisible).
inline int make_divisible(double x, int divisor = 8) {
    return static_cast<int>(std::ceil(x / divisor) * divisor);
}

// Conv = Conv2d(bias=false) -> BatchNorm2d -> SiLU
struct ConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    ConvImpl(int c1, int c2, int k = 1, int s = 1, int p = -1, int g = 1) {
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(c1, c2, k).stride(s)
                .padding(autopad(k, p)).groups(g).bias(false)));
        bn = register_module("bn", torch::nn::BatchNorm2d(c2));
    }
    torch::Tensor forward(torch::Tensor x) { return torch::silu(bn(conv(x))); }
};
TORCH_MODULE(Conv);

// Standard residual bottleneck.
struct BottleneckImpl : torch::nn::Module {
    Conv cv1{nullptr}, cv2{nullptr};
    bool add{false};
    BottleneckImpl(int c1, int c2, bool shortcut = true, double e = 0.5) {
        int c_ = static_cast<int>(c2 * e);
        cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
        cv2 = register_module("cv2", Conv(c_, c2, 3, 1));
        add = shortcut && (c1 == c2);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto y = cv2(cv1(x));
        return add ? x + y : y;
    }
};
TORCH_MODULE(Bottleneck);

// CSP bottleneck with 3 convolutions.
struct C3Impl : torch::nn::Module {
    Conv cv1{nullptr}, cv2{nullptr}, cv3{nullptr};
    torch::nn::Sequential m;
    C3Impl(int c1, int c2, int n = 1, bool shortcut = true, double e = 0.5) {
        int c_ = static_cast<int>(c2 * e);
        cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
        cv2 = register_module("cv2", Conv(c1, c_, 1, 1));
        cv3 = register_module("cv3", Conv(2 * c_, c2, 1, 1));
        for (int i = 0; i < n; ++i)
            m->push_back(Bottleneck(c_, c_, shortcut, 1.0));
        register_module("m", m);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto y1 = m->forward(cv1(x));
        auto y2 = cv2(x);
        return cv3(torch::cat({y1, y2}, 1));
    }
};
TORCH_MODULE(C3);

// Spatial Pyramid Pooling - Fast.
struct SPPFImpl : torch::nn::Module {
    Conv cv1{nullptr}, cv2{nullptr};
    int k;
    SPPFImpl(int c1, int c2, int k_ = 5) : k(k_) {
        int c_ = c1 / 2;
        cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
        cv2 = register_module("cv2", Conv(c_ * 4, c2, 1, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = cv1(x);
        auto y1 = torch::max_pool2d(x, k, 1, k / 2);
        auto y2 = torch::max_pool2d(y1, k, 1, k / 2);
        auto y3 = torch::max_pool2d(y2, k, 1, k / 2);
        return cv2(torch::cat({x, y1, y2, y3}, 1));
    }
};
TORCH_MODULE(SPPF);

}  // namespace yolo
