#pragma once
// A compact YOLO-style detector built entirely on the self-made autograd.
// Now with BatchNorm and a 2-scale FPN head (strides 8 and 16), same *ideas*
// as YOLOv5, shrunk to something a naive-CPU autograd can train.
#include "autograd.h"
#include <vector>
#include <string>

namespace scratch {

struct Box { int cls; float cx, cy, w, h; };  // normalized [0,1]

// Dataset loaded from disk (PPM images + YOLO txt labels), letterboxed to SxS.
struct Dataset {
    int S = 64;
    std::vector<std::vector<float>> imgs;   // each 3*S*S, row-major CHW
    std::vector<std::vector<Box>> boxes;
    size_t size() const { return imgs.size(); }
};
Dataset load_disk_dataset(const std::string& images_dir, int S);

// Conv -> BatchNorm -> SiLU block (conv has no bias; BN's beta replaces it).
struct ConvBN {
    ag::Tensor w, bias;               // bias is a fixed zero (no grad)
    ag::Tensor gamma, beta;           // BN affine params (learnable)
    ag::Tensor run_mean, run_var;     // BN running stats (no grad)
    int stride, pad;
};

// YOLOv5 building blocks (built from ConvBN), for the C3/SPPF backbone.
struct Bottleneck { ConvBN cv1, cv2; };               // residual 1x1 -> 3x3
struct C3 { ConvBN cv1, cv2, cv3; std::vector<Bottleneck> m; };  // CSP block
struct SPPF { ConvBN cv1, cv2; int k = 5; };          // spatial pyramid pooling

struct SmallYolo {
    int S = 64;
    int nc = 3;
    int no;                           // 5 + nc
    int na = 3;                       // anchors per scale (v5-style)
    bool training = true;
    std::vector<int> grids;           // {S/8, S/16}: stride-8 and stride-16 grids
    std::vector<int> strides;         // {8, 16}
    // per-scale anchors in NORMALIZED (w,h), flattened [w0,h0,w1,h1,w2,h2]
    std::vector<std::vector<float>> anchors;

    // v5-like backbone: stride-2 convs + C3 blocks + SPPF, then a 2-scale FPN.
    ConvBN conv0, conv1, conv2, conv3;   // downsampling stem (stride 2)
    C3 c3a, c3b, c3c, c3h;               // CSP blocks (c3b=stride8, c3h=FPN head)
    SPPF sppf;                           // -> stride16 feature
    ag::Tensor h8w, h8b;                 // detection head @ stride 8 (na*no ch)
    ag::Tensor h16w, h16b;               // detection head @ stride 16

    SmallYolo(int classes = 3, int input_size = 64);
    std::vector<ag::Tensor> forward(const ag::Tensor& x);  // one (N,no,G,G) per scale
    std::vector<ag::Tensor> params();
};

// Synthetic in-memory data (colored rectangles) + boxes.
void gen_batch(int N, int S, ag::Tensor& imgs, std::vector<std::vector<Box>>& boxes);

// Multi-scale YOLO-style loss (objectness + box + class), summed over scales.
ag::Tensor compute_loss(SmallYolo& net, const std::vector<ag::Tensor>& preds,
                        const std::vector<std::vector<Box>>& boxes);

// Decode all scales for image n into detections above obj threshold.
std::vector<Box> decode(SmallYolo& net, const std::vector<ag::Tensor>& preds, int n,
                        float conf_thres, std::vector<float>* confs = nullptr);

// data_dir empty => synthetic (nc=3, 64px); else train on the PPM dataset at
// data_dir with the given class count and input size.
int train_demo(const std::string& data_dir = "", int nc = 3, int input_size = 64);
}  // namespace scratch
