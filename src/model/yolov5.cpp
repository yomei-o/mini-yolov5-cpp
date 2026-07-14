#include "model/yolov5.h"
#include <algorithm>

namespace yolo {

// --------------------------- Detect ----------------------------------------
DetectImpl::DetectImpl(int nc_, const std::vector<std::vector<float>>& anchors_,
                       const std::vector<int>& ch) {
    nc = nc_;
    no = nc + 5;
    nl = static_cast<int>(anchors_.size());
    na = static_cast<int>(anchors_[0].size()) / 2;

    auto a = torch::zeros({nl, na, 2});
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < na; ++j) {
            a[i][j][0] = anchors_[i][2 * j];
            a[i][j][1] = anchors_[i][2 * j + 1];
        }
    anchors = register_buffer("anchors", a);

    for (int i = 0; i < nl; ++i)
        m->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(ch[i], no * na, 1)));
    register_module("m", m);
    stride.assign(nl, 0.f);
}

std::vector<torch::Tensor> DetectImpl::forward(std::vector<torch::Tensor> x) {
    std::vector<torch::Tensor> out(nl);
    for (int i = 0; i < nl; ++i) {
        auto conv = m[i]->as<torch::nn::Conv2d>();
        auto z = conv->forward(x[i]);             // (bs, no*na, ny, nx)
        int64_t bs = z.size(0), ny = z.size(2), nx = z.size(3);
        // (bs, na, no, ny, nx) -> (bs, na, ny, nx, no)
        z = z.view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();
        out[i] = z;
    }
    return out;
}

torch::Tensor DetectImpl::decode_inference(const std::vector<torch::Tensor>& preds) {
    std::vector<torch::Tensor> zs;
    for (int i = 0; i < nl; ++i) {
        auto p = preds[i];                        // (bs, na, ny, nx, no)
        int64_t bs = p.size(0), ny = p.size(2), nx = p.size(3);

        // build grid of cell coordinates
        auto yv = torch::arange(ny);
        auto xv = torch::arange(nx);
        auto grids = torch::meshgrid({yv, xv}, "ij");
        auto grid = torch::stack({grids[1], grids[0]}, 2)   // (ny, nx, 2) = (x, y)
                        .view({1, 1, ny, nx, 2}).to(p.dtype());
        auto anchor_grid = (anchors[i] * stride[i])
                               .view({1, na, 1, 1, 2}).to(p.dtype());

        auto y = p.sigmoid();
        auto xy = (y.slice(4, 0, 2) * 2 - 0.5 + grid) * stride[i];
        auto wh = (y.slice(4, 2, 4) * 2).pow(2) * anchor_grid;
        auto conf = y.slice(4, 4, no);
        auto z = torch::cat({xy, wh, conf}, 4);
        zs.push_back(z.view({bs, na * ny * nx, no}));
    }
    return torch::cat(zs, 1);                      // (bs, n, no)
}

// --------------------------- YoloV5 ----------------------------------------
YoloV5Impl::YoloV5Impl(int nc_, double w, double d,
                       const std::vector<std::vector<float>>& anchors) {
    nc = nc_;
    auto cw = [&](int base) { return make_divisible(base * w, 8); };
    auto dn = [&](int base) { return std::max(1, static_cast<int>(std::round(base * d))); };

    // ---- backbone ----
    conv0 = register_module("conv0", Conv(3, cw(64), 6, 2, 2));
    conv1 = register_module("conv1", Conv(cw(64), cw(128), 3, 2));
    c3_2  = register_module("c3_2",  C3(cw(128), cw(128), dn(3)));
    conv3 = register_module("conv3", Conv(cw(128), cw(256), 3, 2));
    c3_4  = register_module("c3_4",  C3(cw(256), cw(256), dn(6)));   // -> P3 skip
    conv5 = register_module("conv5", Conv(cw(256), cw(512), 3, 2));
    c3_6  = register_module("c3_6",  C3(cw(512), cw(512), dn(9)));   // -> P4 skip
    conv7 = register_module("conv7", Conv(cw(512), cw(1024), 3, 2));
    c3_8  = register_module("c3_8",  C3(cw(1024), cw(1024), dn(3)));
    sppf9 = register_module("sppf9", SPPF(cw(1024), cw(1024), 5));

    // ---- head (PAN) ----
    conv10 = register_module("conv10", Conv(cw(1024), cw(512), 1, 1));
    c3_13  = register_module("c3_13",  C3(cw(512) + cw(512), cw(512), dn(3), false));
    conv14 = register_module("conv14", Conv(cw(512), cw(256), 1, 1));
    c3_17  = register_module("c3_17",  C3(cw(256) + cw(256), cw(256), dn(3), false)); // P3 out
    conv18 = register_module("conv18", Conv(cw(256), cw(256), 3, 2));
    c3_20  = register_module("c3_20",  C3(cw(256) + cw(256), cw(512), dn(3), false)); // P4 out
    conv21 = register_module("conv21", Conv(cw(512), cw(512), 3, 2));
    c3_23  = register_module("c3_23",  C3(cw(512) + cw(512), cw(1024), dn(3), false)); // P5 out

    std::vector<int> ch = {cw(256), cw(512), cw(1024)};
    detect = register_module("detect", Detect(nc, anchors, ch));

    set_stride();
}

std::vector<torch::Tensor> YoloV5Impl::forward(torch::Tensor x) {
    // backbone
    x = conv0(x);
    x = conv1(x);
    x = c3_2(x);
    x = conv3(x);
    auto p3 = c3_4(x);         // save
    x = conv5(p3);
    auto p4 = c3_6(x);         // save
    x = conv7(p4);
    x = c3_8(x);
    x = sppf9(x);

    // head
    auto h10 = conv10(x);      // save
    x = torch::upsample_nearest2d(h10, {}, std::vector<double>{2.0, 2.0});
    x = torch::cat({x, p4}, 1);
    x = c3_13(x);

    auto h14 = conv14(x);      // save
    x = torch::upsample_nearest2d(h14, {}, std::vector<double>{2.0, 2.0});
    x = torch::cat({x, p3}, 1);
    auto out_p3 = c3_17(x);    // -> detect layer 0

    x = conv18(out_p3);
    x = torch::cat({x, h14}, 1);
    auto out_p4 = c3_20(x);    // -> detect layer 1

    x = conv21(out_p4);
    x = torch::cat({x, h10}, 1);
    auto out_p5 = c3_23(x);    // -> detect layer 2

    return detect->forward({out_p3, out_p4, out_p5});
}

void YoloV5Impl::set_stride() {
    torch::NoGradGuard ng;
    bool was_training = is_training();
    eval();
    int s = 256;
    auto dummy = torch::zeros({1, 3, s, s});
    auto outs = forward(dummy);
    for (int i = 0; i < detect->nl; ++i) {
        int64_t ny = outs[i].size(2);
        detect->stride[i] = static_cast<float>(s) / static_cast<float>(ny);
    }
    // anchors are stored in pixel units; convert to grid units.
    for (int i = 0; i < detect->nl; ++i)
        detect->anchors[i] = detect->anchors[i] / detect->stride[i];
    if (was_training) train();
}

}  // namespace yolo
