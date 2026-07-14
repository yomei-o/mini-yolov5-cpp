#define _USE_MATH_DEFINES
#include "engine.h"
#include "model/yolov5.h"
#include "loss/loss.h"
#include "data/dataset.h"

#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>

namespace fs = std::filesystem;

namespace yolo {

// Exponential moving average of weights (parameters + buffers), YOLOv5-style.
struct ModelEMA {
    std::vector<torch::Tensor> shadow;
    double decay;
    long updates = 0;
    ModelEMA(YoloV5& model, double d = 0.9999) : decay(d) {
        torch::NoGradGuard ng;
        for (auto& p : model->parameters()) shadow.push_back(p.detach().clone());
        for (auto& b : model->buffers()) shadow.push_back(b.detach().clone());
    }
    void update(YoloV5& model) {
        torch::NoGradGuard ng;
        ++updates;
        double d = decay * (1 - std::exp(-updates / 2000.0));  // ramp-up
        size_t i = 0;
        for (auto& p : model->parameters()) {
            if (p.is_floating_point())
                shadow[i].mul_(d).add_(p.detach(), 1 - d);
            else
                shadow[i].copy_(p.detach());
            ++i;
        }
        for (auto& b : model->buffers()) { shadow[i].copy_(b.detach()); ++i; }
    }
    void copy_to(YoloV5& model) {
        torch::NoGradGuard ng;
        size_t i = 0;
        for (auto& p : model->parameters()) { p.copy_(shadow[i]); ++i; }
        for (auto& b : model->buffers()) { b.copy_(shadow[i]); ++i; }
    }
};

int run_train(const TrainOpts& o) {
    torch::manual_seed(0);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    DatasetConfig dc;
    dc.images_dir = o.images_dir;
    dc.labels_dir = o.labels_dir;
    dc.imgsz = o.imgsz;
    dc.augment = o.augment;
    YoloDataset ds(dc);
    std::cout << "Dataset: " << ds.size() << " images, nc=" << o.nc << "\n";
    if (ds.size() == 0) { std::cerr << "empty dataset\n"; return 1; }

    YoloV5 model(o.nc, 0.25, 0.33);
    if (!o.weights.empty()) {
        std::cout << "Loading weights: " << o.weights << "\n";
        torch::load(model, o.weights);
    }
    model->to(device);

    ComputeLoss criterion(model, o.imgsz);
    ModelEMA ema(model, 0.9999);

    torch::optim::SGD opt(model->parameters(),
                          torch::optim::SGDOptions(o.lr0).momentum(0.937)
                              .weight_decay(o.weight_decay).nesterov(true));

    int batches = (ds.size() + o.batch - 1) / o.batch;
    long total_iters = (long)batches * o.epochs;
    long warmup_iters = std::max<long>(batches, std::min<long>(1000, total_iters / 2));
    double lrf = 0.01;  // final lr fraction

    fs::create_directories(o.out_dir);
    std::vector<size_t> idx(ds.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(0);

    long it = 0;
    for (int epoch = 0; epoch < o.epochs; ++epoch) {
        model->train();
        std::shuffle(idx.begin(), idx.end(), rng);
        double run_loss = 0, run_box = 0, run_obj = 0, run_cls = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (int bi = 0; bi < batches; ++bi) {
            // lr schedule (linear warmup -> cosine decay)
            double lr;
            if (it < warmup_iters) {
                lr = o.lr0 * (double)it / warmup_iters;
            } else {
                double prog = (double)(it - warmup_iters) /
                              std::max<long>(1, total_iters - warmup_iters);
                constexpr double kPi = 3.14159265358979323846;
                lr = o.lr0 * (lrf + (1 - lrf) * 0.5 * (1 + std::cos(kPi * prog)));
            }
            for (auto& g : opt.param_groups())
                static_cast<torch::optim::SGDOptions&>(g.options()).lr(lr);

            // build batch
            std::vector<Sample> samples;
            for (int k = 0; k < o.batch; ++k) {
                size_t s = idx[(bi * o.batch + k) % ds.size()];
                samples.push_back(ds.get(s));
            }
            auto batch = YoloDataset::collate(samples);
            auto imgs = batch.imgs.to(device);
            auto targets = batch.targets.to(device);

            opt.zero_grad();
            auto preds = model->forward(imgs);
            auto r = criterion(preds, targets);
            r.loss.backward();
            opt.step();
            if (o.use_ema) ema.update(model);

            run_loss += r.loss.item<float>();
            run_box += r.lbox.item<float>();
            run_obj += r.lobj.item<float>();
            run_cls += r.lcls.item<float>();
            ++it;
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double cur_lr;
        {
            auto& g = opt.param_groups()[0];
            cur_lr = static_cast<torch::optim::SGDOptions&>(g.options()).lr();
        }
        std::printf("epoch %3d/%d | loss %.4f box %.4f obj %.4f cls %.4f | lr %.5f | %.1fs\n",
                    epoch + 1, o.epochs, run_loss / batches, run_box / batches,
                    run_obj / batches, run_cls / batches, cur_lr, secs);
        std::fflush(stdout);

        // checkpoints
        torch::save(model, (fs::path(o.out_dir) / "last.pt").string());
    }

    // final: save EMA weights as best.pt
    if (o.use_ema) {
        YoloV5 best(o.nc, 0.25, 0.33);
        best->to(device);
        ema.copy_to(best);
        torch::save(best, (fs::path(o.out_dir) / "best.pt").string());
        std::cout << "Saved: " << (fs::path(o.out_dir) / "best.pt").string() << " (EMA)\n";
    }
    std::cout << "Saved: " << (fs::path(o.out_dir) / "last.pt").string() << "\n";
    return 0;
}

}  // namespace yolo
