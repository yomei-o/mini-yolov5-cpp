#include "loss/loss.h"
#include <cmath>

using namespace torch::indexing;

namespace yolo {

torch::Tensor bbox_ciou(const torch::Tensor& box1, const torch::Tensor& box2,
                        double eps) {
    // xywh -> corners
    auto x1 = box1.index({Slice(), 0}), y1 = box1.index({Slice(), 1});
    auto w1 = box1.index({Slice(), 2}), h1 = box1.index({Slice(), 3});
    auto x2 = box2.index({Slice(), 0}), y2 = box2.index({Slice(), 1});
    auto w2 = box2.index({Slice(), 2}), h2 = box2.index({Slice(), 3});

    auto b1x1 = x1 - w1 / 2, b1x2 = x1 + w1 / 2;
    auto b1y1 = y1 - h1 / 2, b1y2 = y1 + h1 / 2;
    auto b2x1 = x2 - w2 / 2, b2x2 = x2 + w2 / 2;
    auto b2y1 = y2 - h2 / 2, b2y2 = y2 + h2 / 2;

    auto inter = (torch::minimum(b1x2, b2x2) - torch::maximum(b1x1, b2x1)).clamp_min(0) *
                 (torch::minimum(b1y2, b2y2) - torch::maximum(b1y1, b2y1)).clamp_min(0);
    auto uni = w1 * h1 + w2 * h2 - inter + eps;
    auto iou = inter / uni;

    auto cw = torch::maximum(b1x2, b2x2) - torch::minimum(b1x1, b2x1);
    auto ch = torch::maximum(b1y2, b2y2) - torch::minimum(b1y1, b2y1);
    auto c2 = cw * cw + ch * ch + eps;
    auto rho2 = ((b2x1 + b2x2 - b1x1 - b1x2).pow(2) +
                 (b2y1 + b2y2 - b1y1 - b1y2).pow(2)) / 4;

    const double pi = 3.14159265358979323846;
    auto v = (4 / (pi * pi)) *
             (torch::atan(w2 / (h2 + eps)) - torch::atan(w1 / (h1 + eps))).pow(2);
    torch::Tensor alpha;
    {
        torch::NoGradGuard ng;
        alpha = v / (v - iou + (1 + eps));
    }
    return iou - (rho2 / c2 + v * alpha);
}

ComputeLoss::ComputeLoss(const YoloV5& model, int imgsz) {
    auto det = model->detect;
    anchors_ = det->anchors.clone();      // (nl,na,2) grid units
    nl_ = det->nl;
    na_ = det->na;
    nc_ = det->nc;
    no_ = det->no;

    balance_ = (nl_ == 3) ? std::vector<double>{4.0, 1.0, 0.4}
                          : std::vector<double>(nl_, 1.0);

    // Hyperparameters (YOLOv5 defaults) with the standard train.py scaling.
    anchor_t_ = 4.0;
    gr_ = 1.0;
    cp_ = 1.0;  // positive class target (no label smoothing)
    cn_ = 0.0;  // negative class target
    box_g_ = 0.05 * 3.0 / nl_;
    cls_g_ = 0.5 * nc_ / 80.0 * 3.0 / nl_;
    obj_g_ = 1.0 * std::pow(imgsz / 640.0, 2) * 3.0 / nl_;

    bce_cls_ = torch::nn::BCEWithLogitsLoss();
    bce_obj_ = torch::nn::BCEWithLogitsLoss();
}

ComputeLoss::Built ComputeLoss::build_targets(
    const std::vector<torch::Tensor>& p, const torch::Tensor& targets) {
    Built out;
    int64_t nt = targets.size(0);
    auto device = targets.device();

    // (na, nt): anchor index broadcast over targets
    auto ai = torch::arange(na_, torch::TensorOptions().dtype(torch::kFloat).device(device))
                  .view({na_, 1}).repeat({1, nt});
    // (na, nt, 7): [img, cls, x, y, w, h, anchor_idx]
    auto t0 = torch::cat({targets.repeat({na_, 1, 1}), ai.unsqueeze(-1)}, 2);

    const double g = 0.5;  // center-cell offset bias
    auto off = torch::tensor({{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f},
                              {-1.f, 0.f}, {0.f, -1.f}},
                             torch::TensorOptions().device(device)) * g;  // (5,2)

    for (int i = 0; i < nl_; ++i) {
        auto anchors_i = anchors_[i].to(device);        // (na,2)
        int64_t ny = p[i].size(2), nx = p[i].size(3);
        auto gain = torch::ones({7}, torch::TensorOptions().device(device));
        gain[2] = (float)nx; gain[3] = (float)ny;
        gain[4] = (float)nx; gain[5] = (float)ny;

        auto t = t0 * gain;                              // (na, nt, 7)
        torch::Tensor offsets;
        if (nt) {
            auto r = t.index({Ellipsis, Slice(4, 6)}) / anchors_i.view({na_, 1, 2});
            auto j = torch::maximum(r, 1.0 / r).amax(2) < anchor_t_;  // (na,nt) bool
            t = t.index({j});                            // (M,7)

            auto gxy = t.index({Slice(), Slice(2, 4)});  // (M,2)
            auto gxi = torch::stack({gain[2], gain[3]}) - gxy;
            auto jk = (gxy.fmod(1) < g) & (gxy > 1.0);   // (M,2)
            auto lm = (gxi.fmod(1) < g) & (gxi > 1.0);
            auto jc = jk.index({Slice(), 0}), kc = jk.index({Slice(), 1});
            auto lc = lm.index({Slice(), 0}), mc = lm.index({Slice(), 1});
            auto ones = torch::ones_like(jc);
            auto mask = torch::stack({ones, jc, kc, lc, mc});   // (5,M)

            t = t.repeat({5, 1, 1}).index({mask});              // (M',7)
            auto off_full = torch::zeros_like(gxy).unsqueeze(0) + off.view({5, 1, 2});
            offsets = off_full.index({mask});                   // (M',2)
        } else {
            t = t0.index({0});                            // (0,7)
            offsets = torch::zeros({0, 2}, torch::TensorOptions().device(device));
        }

        auto b = t.index({Slice(), 0}).to(torch::kLong);
        auto c = t.index({Slice(), 1}).to(torch::kLong);
        auto gxy = t.index({Slice(), Slice(2, 4)});
        auto gwh = t.index({Slice(), Slice(4, 6)});
        auto gij = (gxy - offsets).to(torch::kLong);
        auto gi = gij.index({Slice(), 0}).clamp(0, nx - 1);
        auto gj = gij.index({Slice(), 1}).clamp(0, ny - 1);
        auto a = t.index({Slice(), 6}).to(torch::kLong);

        out.indices.push_back({b, a, gj, gi});
        out.tbox.push_back(torch::cat({gxy - gij.to(gxy.dtype()), gwh}, 1));
        out.anch.push_back(anchors_i.index({a}));
        out.tcls.push_back(c);
    }
    return out;
}

ComputeLoss::Result ComputeLoss::operator()(
    const std::vector<torch::Tensor>& p, const torch::Tensor& targets) {
    auto device = p[0].device();
    auto zero = [&] { return torch::zeros({1}, torch::TensorOptions().device(device)); };
    auto lbox = zero(), lobj = zero(), lcls = zero();

    auto tg = build_targets(p, targets);

    for (int i = 0; i < nl_; ++i) {
        auto pi = p[i];                                   // (bs,na,ny,nx,no)
        auto b = tg.indices[i][0], a = tg.indices[i][1],
             gj = tg.indices[i][2], gi = tg.indices[i][3];
        auto tobj = torch::zeros_like(pi.index({Ellipsis, 0}));  // (bs,na,ny,nx)

        int64_t n = b.size(0);
        if (n) {
            auto ps = pi.index({b, a, gj, gi});           // (n,no)
            auto pxy = ps.index({Slice(), Slice(0, 2)}).sigmoid() * 2 - 0.5;
            auto pwh = (ps.index({Slice(), Slice(2, 4)}).sigmoid() * 2).pow(2) * tg.anch[i];
            auto pbox = torch::cat({pxy, pwh}, 1);

            auto iou = bbox_ciou(pbox, tg.tbox[i]);        // (n,)
            lbox = lbox + (1 - iou).mean();

            auto score = iou.detach().clamp_min(0).to(tobj.dtype());
            tobj.index_put_({b, a, gj, gi}, (1 - gr_) + gr_ * score);

            if (nc_ > 1) {
                auto pcls = ps.index({Slice(), Slice(5, no_)});
                auto tcls = torch::full_like(pcls, cn_);
                tcls.index_put_({torch::arange(n, torch::TensorOptions().device(device)),
                                 tg.tcls[i]}, cp_);
                lcls = lcls + bce_cls_->forward(pcls, tcls);
            }
        }
        auto obji = bce_obj_->forward(pi.index({Ellipsis, 4}), tobj);
        lobj = lobj + obji * balance_[i];
    }

    lbox = lbox * box_g_;
    lobj = lobj * obj_g_;
    lcls = lcls * cls_g_;
    int64_t bs = p[0].size(0);
    auto loss = (lbox + lobj + lcls) * bs;
    return {loss, lbox.detach(), lobj.detach(), lcls.detach()};
}

}  // namespace yolo
