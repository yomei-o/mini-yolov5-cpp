#include "engine.h"
#include "data/image_io.h"

#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace torch::indexing;
namespace fs = std::filesystem;

namespace yolo {

// 3 classes distinguished by color: 0=red, 1=green, 2=blue.
static const float kColors[3][3] = {{0.85f, 0.15f, 0.15f},
                                     {0.15f, 0.80f, 0.20f},
                                     {0.20f, 0.30f, 0.90f}};

static void gen_split(const std::string& img_dir, const std::string& lbl_dir,
                      int count, int S) {
    fs::create_directories(img_dir);
    fs::create_directories(lbl_dir);
    for (int i = 0; i < count; ++i) {
        // background: mid-gray + light noise
        auto img = torch::rand({3, S, S}) * 0.15f + 0.35f;

        int nobj = 1 + (torch::randint(0, 3, {1}).item<int>());  // 1..3 objects
        std::ostringstream label;
        for (int k = 0; k < nobj; ++k) {
            int cls = torch::randint(0, 3, {1}).item<int>();
            int bw = 30 + torch::randint(0, S / 3, {1}).item<int>();
            int bh = 30 + torch::randint(0, S / 3, {1}).item<int>();
            int x1 = torch::randint(0, std::max(1, S - bw), {1}).item<int>();
            int y1 = torch::randint(0, std::max(1, S - bh), {1}).item<int>();
            int x2 = std::min(S - 1, x1 + bw), y2 = std::min(S - 1, y1 + bh);

            // fill rectangle with class color + slight noise
            for (int c = 0; c < 3; ++c) {
                auto patch = torch::rand({y2 - y1, x2 - x1}) * 0.1f + kColors[cls][c];
                img.index_put_({c, Slice(y1, y2), Slice(x1, x2)}, patch.clamp(0, 1));
            }
            // YOLO label: class cx cy w h (normalized)
            float cx = (x1 + x2) / 2.0f / S, cy = (y1 + y2) / 2.0f / S;
            float w = (x2 - x1) / (float)S, h = (y2 - y1) / (float)S;
            label << cls << " " << cx << " " << cy << " " << w << " " << h << "\n";
        }

        std::ostringstream name;
        name << "img_" << std::setw(4) << std::setfill('0') << i;
        save_image_rgb((fs::path(img_dir) / (name.str() + ".png")).string(), img);
        std::ofstream((fs::path(lbl_dir) / (name.str() + ".txt")).string()) << label.str();
    }
}

int run_gen_data(const GenDataOpts& o) {
    torch::manual_seed(1234);
    int S = o.imgsz;
    gen_split((fs::path(o.out_dir) / "images" / "train").string(),
              (fs::path(o.out_dir) / "labels" / "train").string(), o.num_train, S);
    gen_split((fs::path(o.out_dir) / "images" / "val").string(),
              (fs::path(o.out_dir) / "labels" / "val").string(), o.num_val, S);
    std::cout << "Generated synthetic dataset in " << o.out_dir << "\n"
              << "  train: " << o.num_train << " images\n"
              << "  val:   " << o.num_val << " images\n"
              << "  classes: 3 (0=red, 1=green, 2=blue), imgsz=" << S << "\n";
    return 0;
}

}  // namespace yolo
