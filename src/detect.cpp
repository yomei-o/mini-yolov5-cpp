#include "engine.h"
#include "model/yolov5.h"
#include "data/dataset.h"
#include "data/image_io.h"
#include "utils.h"

#include <torch/torch.h>
#include <iostream>

using namespace torch::indexing;

namespace yolo {

int run_detect(const DetectOpts& o) {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    YoloV5 model(o.nc, 0.25, 0.33);
    torch::load(model, o.weights);
    model->to(device);
    model->eval();

    int w0, h0;
    auto orig = load_image_rgb(o.source, w0, h0);
    if (!orig.defined()) { std::cerr << "cannot load " << o.source << "\n"; return 1; }

    auto lb = letterbox(orig, o.imgsz);
    auto input = lb.img.unsqueeze(0).to(device);

    torch::Tensor det;
    {
        torch::NoGradGuard ng;
        auto preds = model->forward(input);
        auto dec = model->detect->decode_inference(preds);   // (1,n,no)
        det = non_max_suppression(dec[0].cpu(), o.conf_thres, o.iou_thres);
    }
    std::cout << "detections: " << det.size(0) << "\n";

    // map boxes from letterboxed space back to original image, draw on canvas
    auto canvas = lb.img.clone();
    const std::array<float, 3> palette[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0.4f, 1}};
    for (int i = 0; i < det.size(0); ++i) {
        float x1 = det[i][0].item<float>(), y1 = det[i][1].item<float>();
        float x2 = det[i][2].item<float>(), y2 = det[i][3].item<float>();
        float conf = det[i][4].item<float>();
        int cls = det[i][5].item<int>();
        // coordinates in original-image pixels (for reporting)
        float ox1 = (x1 - lb.left) / lb.r, oy1 = (y1 - lb.top) / lb.r;
        float ox2 = (x2 - lb.left) / lb.r, oy2 = (y2 - lb.top) / lb.r;
        std::printf("  cls=%d conf=%.3f box=[%.1f %.1f %.1f %.1f]\n",
                    cls, conf, ox1, oy1, ox2, oy2);
        draw_rect(canvas, (int)x1, (int)y1, (int)x2, (int)y2,
                  palette[cls % 3], 2);
    }
    save_image_rgb(o.out, canvas);
    std::cout << "saved: " << o.out << "\n";
    return 0;
}

}  // namespace yolo
