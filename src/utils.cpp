#include "utils.h"
#include <algorithm>

using namespace torch::indexing;

namespace yolo {

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto cx = x.index({Ellipsis, 0}), cy = x.index({Ellipsis, 1});
    auto w = x.index({Ellipsis, 2}), h = x.index({Ellipsis, 3});
    y.index_put_({Ellipsis, 0}, cx - w / 2);
    y.index_put_({Ellipsis, 1}, cy - h / 2);
    y.index_put_({Ellipsis, 2}, cx + w / 2);
    y.index_put_({Ellipsis, 3}, cy + h / 2);
    return y;
}

torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores,
                  double iou_thres) {
    if (boxes.size(0) == 0) return torch::empty({0}, torch::kLong);
    auto x1 = boxes.index({Slice(), 0}), y1 = boxes.index({Slice(), 1});
    auto x2 = boxes.index({Slice(), 2}), y2 = boxes.index({Slice(), 3});
    auto areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0);
    auto order = std::get<1>(scores.sort(0, /*descending=*/true));

    std::vector<int64_t> keep;
    auto order_a = order.accessor<int64_t, 1>();
    std::vector<bool> suppressed(order.size(0), false);
    for (int64_t i = 0; i < order.size(0); ++i) {
        int64_t idx = order_a[i];
        if (suppressed[i]) continue;
        keep.push_back(idx);
        for (int64_t j = i + 1; j < order.size(0); ++j) {
            if (suppressed[j]) continue;
            int64_t jdx = order_a[j];
            auto xx1 = std::max(x1[idx].item<float>(), x1[jdx].item<float>());
            auto yy1 = std::max(y1[idx].item<float>(), y1[jdx].item<float>());
            auto xx2 = std::min(x2[idx].item<float>(), x2[jdx].item<float>());
            auto yy2 = std::min(y2[idx].item<float>(), y2[jdx].item<float>());
            float iw = std::max(0.f, xx2 - xx1), ih = std::max(0.f, yy2 - yy1);
            float inter = iw * ih;
            float uni = areas[idx].item<float>() + areas[jdx].item<float>() - inter;
            if (uni > 0 && inter / uni > iou_thres) suppressed[j] = true;
        }
    }
    return torch::tensor(keep, torch::kLong);
}

torch::Tensor non_max_suppression(const torch::Tensor& pred, double conf_thres,
                                  double iou_thres, int max_det) {
    int no = pred.size(1);
    int nc = no - 5;
    // filter by objectness
    auto mask = pred.index({Slice(), 4}) > conf_thres;
    auto x = pred.index({mask});
    if (x.size(0) == 0) return torch::zeros({0, 6});

    // conf = obj * cls
    auto cls = x.index({Slice(), Slice(5, no)}) * x.index({Slice(), Slice(4, 5)});
    auto box = xywh2xyxy(x.index({Slice(), Slice(0, 4)}));
    auto mx = cls.max(1);
    auto conf = std::get<0>(mx);
    auto j = std::get<1>(mx).to(torch::kFloat32);

    auto keep_conf = conf > conf_thres;
    box = box.index({keep_conf});
    conf = conf.index({keep_conf});
    j = j.index({keep_conf});
    if (box.size(0) == 0) return torch::zeros({0, 6});

    // per-class NMS via large per-class coordinate offset
    float max_wh = 7680.f;
    auto offset = j.unsqueeze(1) * max_wh;
    auto boxes_off = box + offset;
    auto idx = nms(boxes_off, conf, iou_thres);
    if (idx.size(0) > max_det) idx = idx.index({Slice(0, max_det)});

    auto out = torch::cat({box.index({idx}),
                           conf.index({idx}).unsqueeze(1),
                           j.index({idx}).unsqueeze(1)}, 1);
    return out;  // (m,6)
}

void draw_rect(torch::Tensor& img, int x1, int y1, int x2, int y2,
               std::array<float, 3> color, int thickness) {
    int H = img.size(1), W = img.size(2);
    auto clampx = [&](int v) { return std::max(0, std::min(W - 1, v)); };
    auto clampy = [&](int v) { return std::max(0, std::min(H - 1, v)); };
    x1 = clampx(x1); x2 = clampx(x2); y1 = clampy(y1); y2 = clampy(y2);
    for (int c = 0; c < 3; ++c) {
        for (int t = 0; t < thickness; ++t) {
            int yt = clampy(y1 + t), yb = clampy(y2 - t);
            int xl = clampx(x1 + t), xr = clampx(x2 - t);
            img.index_put_({c, yt, Slice(x1, x2 + 1)}, color[c]);
            img.index_put_({c, yb, Slice(x1, x2 + 1)}, color[c]);
            img.index_put_({c, Slice(y1, y2 + 1), xl}, color[c]);
            img.index_put_({c, Slice(y1, y2 + 1), xr}, color[c]);
        }
    }
}

}  // namespace yolo
