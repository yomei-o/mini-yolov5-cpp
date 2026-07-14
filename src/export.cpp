#include "engine.h"
#include "model/yolov5.h"
#include "data/image_io.h"

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <filesystem>
#include <algorithm>

namespace yolo {

// Serialize all floating-point parameters & buffers to a portable binary blob
// that the Python bridge (tools/export_onnx.py) reads to reconstruct an
// identical PyTorch model and run torch.onnx.export.
//
// Format:  "YW01" | int32 count | { int32 name_len, name,
//                                    int32 ndim, int64 dims[ndim],
//                                    float32 data[numel] } * count
int run_export_weights(const std::string& weights, int nc, const std::string& out) {
    YoloV5 model(nc, 0.25, 0.33);
    torch::load(model, weights);
    model->to(torch::kCPU);
    model->eval();

    std::vector<std::pair<std::string, torch::Tensor>> items;
    for (const auto& p : model->named_parameters())
        items.emplace_back(p.key(), p.value());
    for (const auto& b : model->named_buffers())
        items.emplace_back(b.key(), b.value());

    std::ofstream f(out, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << out << "\n"; return 1; }
    f.write("YW01", 4);

    int32_t count = 0;
    for (auto& it : items)
        if (it.second.is_floating_point()) ++count;
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));

    int written = 0;
    for (auto& it : items) {
        auto t = it.second.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (!it.second.is_floating_point()) continue;  // skip num_batches_tracked
        int32_t nlen = static_cast<int32_t>(it.first.size());
        f.write(reinterpret_cast<const char*>(&nlen), sizeof(nlen));
        f.write(it.first.data(), nlen);
        int32_t ndim = static_cast<int32_t>(t.dim());
        f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (int i = 0; i < ndim; ++i) {
            int64_t d = t.size(i);
            f.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }
        f.write(reinterpret_cast<const char*>(t.data_ptr<float>()),
                t.numel() * sizeof(float));
        ++written;
    }
    std::cout << "exported " << written << " tensors to " << out << "\n";
    return 0;
}

// Inverse of export-weights: read a "YW01" blob (e.g. produced by
// tools/convert_ultralytics.py) into the model, then save a normal .pt that
// detect/train can consume. This is how official Ultralytics weights end up
// runnable in C++.
int run_import_weights(const std::string& in, int nc, const std::string& out) {
    std::ifstream f(in, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << in << "\n"; return 1; }
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "YW01") { std::cerr << "bad magic in " << in << "\n"; return 1; }
    int32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));

    std::unordered_map<std::string, torch::Tensor> blob;
    for (int t = 0; t < count; ++t) {
        int32_t nlen = 0; f.read(reinterpret_cast<char*>(&nlen), sizeof(nlen));
        std::string name(nlen, '\0'); f.read(&name[0], nlen);
        int32_t ndim = 0; f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int64_t> dims(ndim);
        int64_t numel = 1;
        for (int i = 0; i < ndim; ++i) {
            f.read(reinterpret_cast<char*>(&dims[i]), sizeof(int64_t));
            numel *= dims[i];
        }
        std::vector<float> data(numel);
        f.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
        blob[name] = torch::from_blob(data.data(), dims, torch::kFloat32).clone();
    }

    YoloV5 model(nc, 0.25, 0.33);
    model->to(torch::kCPU);

    int loaded = 0, missing = 0;
    torch::NoGradGuard ng;
    auto assign = [&](const std::string& key, torch::Tensor& dst) {
        auto it = blob.find(key);
        if (it == blob.end()) { ++missing; return; }
        if (dst.sizes() != it->second.sizes()) {
            std::cerr << "shape mismatch for " << key << "\n"; return;
        }
        dst.copy_(it->second);
        ++loaded;
    };
    for (auto& p : model->named_parameters()) { auto v = p.value(); assign(p.key(), v); }
    for (auto& b : model->named_buffers()) {
        if (!b.value().is_floating_point()) continue;  // skip num_batches_tracked
        auto v = b.value(); assign(b.key(), v);
    }

    std::cout << "imported " << loaded << " tensors (missing " << missing
              << ") from " << in << "\n";
    torch::save(model, out);
    std::cout << "saved: " << out << "\n";
    return 0;
}

// Convert a YOLO dataset's images (PNG/JPG, decoded via stb) into PPM (P6) so
// the dependency-free scratch/ trainer can read the *same* data. Labels are
// copied verbatim (already plain text). Output mirrors images/ + labels/.
int run_to_ppm(const std::string& images_dir, const std::string& labels_dir,
               const std::string& out_dir) {
    namespace fs = std::filesystem;
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp"};
    fs::create_directories(fs::path(out_dir) / "images");
    fs::create_directories(fs::path(out_dir) / "labels");

    auto label_of = [&](const fs::path& img) {
        std::string stem = img.stem().string();
        if (!labels_dir.empty()) return (fs::path(labels_dir) / (stem + ".txt")).string();
        std::string s = img.parent_path().string();
        auto pos = s.rfind("images");
        if (pos != std::string::npos) s.replace(pos, 6, "labels");
        return (fs::path(s) / (stem + ".txt")).string();
    };

    int n = 0;
    for (const auto& e : fs::directory_iterator(images_dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;

        int w, h;
        auto img = load_image_rgb(e.path().string(), w, h);  // (3,H,W) float [0,1]
        if (!img.defined()) continue;
        auto u8 = (img.clamp(0, 1) * 255).to(torch::kUInt8)
                      .permute({1, 2, 0}).contiguous();       // (H,W,3)
        std::string stem = e.path().stem().string();
        std::ofstream out((fs::path(out_dir) / "images" / (stem + ".ppm")).string(),
                          std::ios::binary);
        out << "P6\n" << w << " " << h << "\n255\n";
        out.write(reinterpret_cast<const char*>(u8.data_ptr<uint8_t>()), (size_t)w * h * 3);

        // copy label if present
        std::ifstream lin(label_of(e.path()), std::ios::binary);
        std::ofstream lout((fs::path(out_dir) / "labels" / (stem + ".txt")).string());
        if (lin) lout << lin.rdbuf();
        ++n;
    }
    std::cout << "converted " << n << " images to PPM in " << out_dir << "\n";
    return 0;
}

}  // namespace yolo
