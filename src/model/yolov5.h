#pragma once
#include <torch/torch.h>
#include <vector>
#include "model/modules.h"

namespace yolo {

// Detection head. Produces, per detection layer, a tensor shaped
// (batch, na, ny, nx, no) where no = nc + 5. In training mode we return these
// raw grids (what the loss consumes); decode_inference() turns them into
// absolute xywh + scores for NMS.
struct DetectImpl : torch::nn::Module {
    int nc, no, nl, na;
    torch::Tensor anchors;            // (nl, na, 2), in *grid* units after set_stride
    torch::nn::ModuleList m;          // 1x1 conv per layer
    std::vector<float> stride;        // pixels per grid cell, per layer

    DetectImpl(int nc_, const std::vector<std::vector<float>>& anchors_,
               const std::vector<int>& ch);

    // returns nl tensors of shape (bs, na, ny, nx, no)
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

    // Decode raw grids into (bs, n, no) with box as xywh in pixels.
    torch::Tensor decode_inference(const std::vector<torch::Tensor>& preds);
};
TORCH_MODULE(Detect);

// Full YOLOv5 model (backbone + PAN head + Detect).
// width/depth multiples select the variant: n=(0.25,0.33), s=(0.5,0.33),
// m=(0.75,0.67), l=(1.0,1.0), x=(1.25,1.33).
struct YoloV5Impl : torch::nn::Module {
    int nc;
    Conv   conv0{nullptr}, conv1{nullptr}, conv3{nullptr}, conv5{nullptr},
           conv7{nullptr}, conv10{nullptr}, conv14{nullptr}, conv18{nullptr},
           conv21{nullptr};
    C3     c3_2{nullptr}, c3_4{nullptr}, c3_6{nullptr}, c3_8{nullptr},
           c3_13{nullptr}, c3_17{nullptr}, c3_20{nullptr}, c3_23{nullptr};
    SPPF   sppf9{nullptr};
    Detect detect{nullptr};

    YoloV5Impl(int nc_ = 80, double width_multiple = 0.25,
               double depth_multiple = 0.33,
               const std::vector<std::vector<float>>& anchors = default_anchors());

    // Training/loss view: returns the Detect raw grids.
    std::vector<torch::Tensor> forward(torch::Tensor x);

    static std::vector<std::vector<float>> default_anchors() {
        // P3/8, P4/16, P5/32 (COCO defaults)
        return {{10, 13, 16, 30, 33, 23},
                {30, 61, 62, 45, 59, 119},
                {116, 90, 156, 198, 373, 326}};
    }

private:
    // Compute detection strides from a dummy forward pass and scale anchors.
    void set_stride();
};
TORCH_MODULE(YoloV5);

}  // namespace yolo
