#include <torch/torch.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include "engine.h"

namespace {

// Parse "--key value" and "--flag" style args into a map.
std::unordered_map<std::string, std::string> parse(int argc, char** argv, int start) {
    std::unordered_map<std::string, std::string> m;
    for (int i = start; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            std::string key = a.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                m[key] = argv[++i];
            } else {
                m[key] = "1";  // boolean flag
            }
        }
    }
    return m;
}

std::string get(const std::unordered_map<std::string, std::string>& m,
                const std::string& k, const std::string& def) {
    auto it = m.find(k);
    return it == m.end() ? def : it->second;
}

void usage() {
    std::cout <<
        "yolov5_cpp - YOLOv5 training/inference in C++ (LibTorch)\n\n"
        "Usage:\n"
        "  yolov5 gen-data [--out DIR] [--num-train N] [--num-val N] [--imgsz S]\n"
        "  yolov5 train --images DIR [--labels DIR] --nc N [--epochs E] [--batch B]\n"
        "               [--imgsz S] [--lr0 L] [--out DIR] [--no-aug] [--no-ema] [--weights W]\n"
        "  yolov5 detect --weights W --source IMG --nc N [--imgsz S]\n"
        "               [--conf C] [--iou I] [--out IMG]\n"
        "  yolov5 export-weights --weights W --nc N [--out weights.bin]\n"
        "               (then: python tools/export_onnx.py --weights weights.bin --nc N --out model.onnx)\n"
        "  yolov5 import-weights --in weights.bin --nc N [--out imported.pt]\n"
        "               (convert official: python tools/convert_ultralytics.py --pt yolov5n.pt --out weights.bin)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    auto a = parse(argc, argv, 2);

    if (cmd == "gen-data") {
        yolo::GenDataOpts o;
        o.out_dir = get(a, "out", o.out_dir);
        o.num_train = std::stoi(get(a, "num-train", "64"));
        o.num_val = std::stoi(get(a, "num-val", "16"));
        o.imgsz = std::stoi(get(a, "imgsz", "320"));
        return yolo::run_gen_data(o);
    }
    if (cmd == "train") {
        yolo::TrainOpts o;
        o.images_dir = get(a, "images", "");
        o.labels_dir = get(a, "labels", "");
        o.nc = std::stoi(get(a, "nc", "80"));
        o.epochs = std::stoi(get(a, "epochs", "100"));
        o.batch = std::stoi(get(a, "batch", "16"));
        o.imgsz = std::stoi(get(a, "imgsz", "640"));
        o.lr0 = std::stod(get(a, "lr0", "0.01"));
        o.out_dir = get(a, "out", "runs");
        o.augment = a.find("no-aug") == a.end();
        o.use_ema = a.find("no-ema") == a.end();
        o.weights = get(a, "weights", "");
        if (o.images_dir.empty()) { std::cerr << "--images required\n"; return 1; }
        return yolo::run_train(o);
    }
    if (cmd == "detect") {
        yolo::DetectOpts o;
        o.weights = get(a, "weights", "");
        o.source = get(a, "source", "");
        o.nc = std::stoi(get(a, "nc", "80"));
        o.imgsz = std::stoi(get(a, "imgsz", "640"));
        o.conf_thres = std::stod(get(a, "conf", "0.25"));
        o.iou_thres = std::stod(get(a, "iou", "0.45"));
        o.out = get(a, "out", "detect_out.png");
        if (o.weights.empty() || o.source.empty()) {
            std::cerr << "--weights and --source required\n"; return 1;
        }
        return yolo::run_detect(o);
    }

    if (cmd == "export-weights") {
        std::string w = get(a, "weights", "");
        int nc = std::stoi(get(a, "nc", "80"));
        std::string out = get(a, "out", "weights.bin");
        if (w.empty()) { std::cerr << "--weights required\n"; return 1; }
        return yolo::run_export_weights(w, nc, out);
    }

    if (cmd == "import-weights") {
        std::string in = get(a, "in", "weights.bin");
        int nc = std::stoi(get(a, "nc", "80"));
        std::string out = get(a, "out", "imported.pt");
        return yolo::run_import_weights(in, nc, out);
    }

    if (cmd == "to-ppm") {
        std::string images = get(a, "images", "");
        std::string labels = get(a, "labels", "");
        std::string out = get(a, "out", "data_ppm");
        if (images.empty()) { std::cerr << "--images required\n"; return 1; }
        return yolo::run_to_ppm(images, labels, out);
    }

    usage();
    return 1;
}
