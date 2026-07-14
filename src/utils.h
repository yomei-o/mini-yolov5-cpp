#pragma once
#include <torch/torch.h>

namespace yolo {

// (..,4) xywh (center) -> xyxy (corners).
torch::Tensor xywh2xyxy(const torch::Tensor& x);

// Greedy IoU NMS. boxes (n,4) xyxy, scores (n,). Returns kept indices (Long).
torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores,
                  double iou_thres);

// YOLOv5-style post-processing for a single image's decoded predictions.
// pred: (n, no) = [cx,cy,w,h,obj,cls...] in pixel units.
// Returns (m,6): x1,y1,x2,y2,conf,class.
torch::Tensor non_max_suppression(const torch::Tensor& pred,
                                  double conf_thres = 0.25,
                                  double iou_thres = 0.45,
                                  int max_det = 300);

// Draw a rectangle outline (thickness px) into a (3,H,W) float [0,1] image.
void draw_rect(torch::Tensor& img, int x1, int y1, int x2, int y2,
               std::array<float, 3> color, int thickness = 2);

}  // namespace yolo
