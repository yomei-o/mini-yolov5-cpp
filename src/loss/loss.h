#pragma once
#include <torch/torch.h>
#include <vector>
#include "model/yolov5.h"

namespace yolo {

// Complete Intersection-over-Union between two sets of xywh boxes (n,4).
// Returns (n,) CIoU values. Used both as a loss term and to weight objectness.
torch::Tensor bbox_ciou(const torch::Tensor& box1, const torch::Tensor& box2,
                        double eps = 1e-7);

// Port of YOLOv5 ComputeLoss. Consumes the model's raw grid outputs and the
// per-image targets, returns the scalar training loss plus its 3 components.
class ComputeLoss {
public:
    // imgsz is used only to scale the objectness gain, matching train.py.
    ComputeLoss(const YoloV5& model, int imgsz = 640);

    struct Result {
        torch::Tensor loss;                 // scalar, already * batch_size
        torch::Tensor lbox, lobj, lcls;     // components (for logging)
    };

    // preds: nl tensors (bs,na,ny,nx,no); targets: (nt,6)=[img,cls,x,y,w,h] norm.
    Result operator()(const std::vector<torch::Tensor>& preds,
                      const torch::Tensor& targets);

private:
    struct Built {
        std::vector<std::array<torch::Tensor, 4>> indices;  // b,a,gj,gi
        std::vector<torch::Tensor> tbox, anch, tcls;
    };
    Built build_targets(const std::vector<torch::Tensor>& preds,
                        const torch::Tensor& targets);

    torch::Tensor anchors_;        // (nl, na, 2) grid units
    int nl_, na_, nc_, no_;
    std::vector<double> balance_;
    double box_g_, obj_g_, cls_g_, anchor_t_, cp_, cn_, gr_;
    torch::nn::BCEWithLogitsLoss bce_cls_{nullptr}, bce_obj_{nullptr};
};

}  // namespace yolo
