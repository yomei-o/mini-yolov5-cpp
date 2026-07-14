#pragma once
#include <string>
#include <vector>

// High-level entry points wired up from main.cpp's CLI.
namespace yolo {

struct TrainOpts {
    std::string images_dir;
    std::string labels_dir;     // optional
    std::string out_dir = "runs";
    int nc = 80;
    int epochs = 100;
    int batch = 16;
    int imgsz = 640;
    double lr0 = 0.01;
    double weight_decay = 5e-4;
    bool augment = true;
    bool use_ema = true;
    std::string weights;        // optional: resume/finetune from a checkpoint
};

struct DetectOpts {
    std::string weights;
    std::string source;         // image file
    std::string out = "detect_out.png";
    int nc = 80;
    int imgsz = 640;
    double conf_thres = 0.25;
    double iou_thres = 0.45;
};

struct GenDataOpts {
    std::string out_dir = "data/synthetic";
    int num_train = 64;
    int num_val = 16;
    int imgsz = 320;
};

int run_train(const TrainOpts& o);
int run_detect(const DetectOpts& o);
int run_gen_data(const GenDataOpts& o);
int run_export_weights(const std::string& weights, int nc, const std::string& out);
int run_import_weights(const std::string& in, int nc, const std::string& out);
int run_to_ppm(const std::string& images_dir, const std::string& labels_dir,
               const std::string& out_dir);

}  // namespace yolo
