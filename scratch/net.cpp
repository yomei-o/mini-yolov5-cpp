#include "net.h"
#include "stb_image.h"        // declarations only; implementation in stb_impl.cpp
#include "stb_image_write.h"  // declarations only; implementation in stb_impl.cpp
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdint>
#include <chrono>

using namespace ag;
namespace fs = std::filesystem;

namespace scratch {

// --- pure-C++ PPM (P6) reader (no image library) ---------------------------
static bool load_ppm(const std::string& path, std::vector<float>& chw, int& W, int& H) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic; f >> magic;
    if (magic != "P6") return false;
    int maxv;
    f >> W >> H >> maxv;
    f.get();  // single whitespace after header
    std::vector<unsigned char> buf((size_t)W * H * 3);
    f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    if (!f) return false;
    // interleaved RGB (HWC) -> planar CHW float [0,1]
    chw.assign((size_t)3 * W * H, 0.f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                chw[(c * H + y) * W + x] = buf[(y * W + x) * 3 + c] / 255.f;
    return true;
}

// Read any image: PPM via the pure loader above; JPG/PNG/BMP via stb_image.
// Returns planar CHW float [0,1], RGB.
static bool load_image_any(const std::string& path, std::vector<float>& chw,
                           int& W, int& H) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".ppm") return load_ppm(path, chw, W, H);
    int w, h, c;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 3);  // force RGB
    if (!d) return false;
    W = w; H = h;
    chw.assign((size_t)3 * W * H, 0.f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int k = 0; k < 3; ++k)
                chw[(k * H + y) * W + x] = d[(y * W + x) * 3 + k] / 255.f;
    stbi_image_free(d);
    return true;
}

// Nearest-neighbour letterbox of a (3,H,W) image into (3,S,S); remaps boxes.
static void letterbox_to(const std::vector<float>& src, int W, int H, int S,
                         std::vector<float>& dst, std::vector<Box>& boxes) {
    float r = std::min((float)S / W, (float)S / H);
    int nw = (int)std::round(W * r), nh = (int)std::round(H * r);
    int padx = (S - nw) / 2, pady = (S - nh) / 2;
    dst.assign((size_t)3 * S * S, 114.f / 255.f);  // gray pad
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x) {
                int sx = std::min(W - 1, (int)(x / r)), sy = std::min(H - 1, (int)(y / r));
                dst[(c * S + (pady + y)) * S + (padx + x)] = src[(c * H + sy) * W + sx];
            }
    for (auto& b : boxes) {
        b.cx = (b.cx * W * r + padx) / S;
        b.cy = (b.cy * H * r + pady) / S;
        b.w = b.w * W * r / S;
        b.h = b.h * H * r / S;
    }
}

Dataset load_disk_dataset(const std::string& images_dir, int S) {
    Dataset ds; ds.S = S;
    static const std::vector<std::string> exts = {".ppm", ".jpg", ".jpeg", ".png", ".bmp"};
    // resolve images dir: accept either <dir> or <dir>/images
    fs::path idir = images_dir;
    if (fs::exists(idir / "images")) idir = idir / "images";
    for (const auto& e : fs::directory_iterator(idir)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;
        std::vector<float> rgb; int W, H;
        if (!load_image_any(e.path().string(), rgb, W, H)) continue;
        // label path: swap "images" segment for "labels", extension -> .txt
        fs::path lp = e.path();
        std::string s = lp.parent_path().string();
        auto pos = s.rfind("images");
        if (pos != std::string::npos) s.replace(pos, 6, "labels");
        std::string label = (fs::path(s) / (lp.stem().string() + ".txt")).string();
        std::vector<Box> boxes;
        std::ifstream lin(label);
        std::string line;
        while (std::getline(lin, line)) {
            std::istringstream ss(line);
            Box b; float c;
            if (ss >> c >> b.cx >> b.cy >> b.w >> b.h) { b.cls = (int)c; boxes.push_back(b); }
        }
        std::vector<float> img;
        letterbox_to(rgb, W, H, S, img, boxes);
        ds.imgs.push_back(std::move(img));
        ds.boxes.push_back(std::move(boxes));
    }
    return ds;
}

// Assemble a minibatch (random if shuffle, else the first N) from a Dataset.
static void batch_from(const Dataset& ds, int N, Tensor& imgs,
                       std::vector<std::vector<Box>>& boxes, bool shuffle) {
    int S = ds.S;
    imgs = Tensor::zeros({N, 3, S, S}, false);
    boxes.assign(N, {});
    for (int k = 0; k < N; ++k) {
        int idx = shuffle ? (int)(randf() * ds.size()) % (int)ds.size() : k % (int)ds.size();
        std::copy(ds.imgs[idx].begin(), ds.imgs[idx].end(),
                  imgs.data().begin() + (size_t)k * 3 * S * S);
        boxes[k] = ds.boxes[idx];
    }
}

static Tensor ones(std::vector<int> shape, bool rg) {
    auto t = Tensor::zeros(shape, rg);
    for (auto& v : t.data()) v = 1.f;
    return t;
}

// Conv(no bias) -> BatchNorm -> [SiLU]. He-style weight init.
static ConvBN make_convbn(int cin, int cout, int k, int stride, int pad) {
    ConvBN c;
    c.w = Tensor::randn({cout, cin, k, k}, std::sqrt(2.0f / (cin * k * k)), true);
    c.bias = Tensor::zeros({cout}, false);   // fixed zero; BN.beta does the shift
    c.gamma = ones({cout}, true);
    c.beta = Tensor::zeros({cout}, true);
    c.run_mean = Tensor::zeros({cout}, false);
    c.run_var = ones({cout}, false);
    c.stride = stride; c.pad = pad;
    return c;
}

static Tensor block(SmallYolo& net, ConvBN& c, const Tensor& x) {
    auto h = conv2d(x, c.w, c.bias, c.stride, c.pad);
    h = batchnorm2d(h, c.gamma, c.beta, c.run_mean, c.run_var, net.training);
    return silu(h);
}

static Tensor head_init(int cin, int cout, ag::Tensor& w, ag::Tensor& b) {
    w = Tensor::randn({cout, cin, 1, 1}, std::sqrt(2.0f / cin), true);
    b = Tensor::zeros({cout}, true);
    return w;
}

// ---- C3 / SPPF builders & forwards -----------------------------------------
static C3 make_c3(int cin, int cout, int n) {
    int c_ = cout / 2;
    C3 m;
    m.cv1 = make_convbn(cin, c_, 1, 1, 0);
    m.cv2 = make_convbn(cin, c_, 1, 1, 0);
    m.cv3 = make_convbn(2 * c_, cout, 1, 1, 0);
    for (int i = 0; i < n; ++i) {
        Bottleneck b;
        b.cv1 = make_convbn(c_, c_, 1, 1, 0);
        b.cv2 = make_convbn(c_, c_, 3, 1, 1);
        m.m.push_back(b);
    }
    return m;
}
static SPPF make_sppf(int cin, int cout, int k = 5) {
    SPPF s; s.k = k;
    int c_ = cin / 2;
    s.cv1 = make_convbn(cin, c_, 1, 1, 0);
    s.cv2 = make_convbn(4 * c_, cout, 1, 1, 0);
    return s;
}
static Tensor fwd_c3(SmallYolo& net, C3& m, const Tensor& x) {
    auto y = block(net, m.cv1, x);
    for (auto& b : m.m) y = add(y, block(net, b.cv2, block(net, b.cv1, y)));  // residual
    auto y2 = block(net, m.cv2, x);
    return block(net, m.cv3, cat_channels({y, y2}));
}
static Tensor fwd_sppf(SmallYolo& net, SPPF& s, const Tensor& x) {
    auto x1 = block(net, s.cv1, x);
    auto p1 = maxpool2d(x1, s.k, 1, s.k / 2);
    auto p2 = maxpool2d(p1, s.k, 1, s.k / 2);
    auto p3 = maxpool2d(p2, s.k, 1, s.k / 2);
    return block(net, s.cv2, cat_channels({x1, p1, p2, p3}));
}
static void collect(ConvBN& c, std::vector<Tensor>& p) {
    p.push_back(c.w); p.push_back(c.gamma); p.push_back(c.beta);
}
static void collect_c3(C3& m, std::vector<Tensor>& p) {
    collect(m.cv1, p); collect(m.cv2, p); collect(m.cv3, p);
    for (auto& b : m.m) { collect(b.cv1, p); collect(b.cv2, p); }
}

SmallYolo::SmallYolo(int classes, int input_size) : S(input_size), nc(classes) {
    no = 5 + nc;
    grids = {S / 8, S / 16};
    strides = {8, 16};
    anchors = {{0.05f, 0.08f, 0.10f, 0.16f, 0.18f, 0.28f},   // stride 8
               {0.25f, 0.30f, 0.40f, 0.50f, 0.65f, 0.80f}};  // stride 16
    // v5-like: stride-2 conv downsampling + C3 blocks + SPPF
    conv0 = make_convbn(3, 16, 3, 2, 1);    // /2
    conv1 = make_convbn(16, 32, 3, 2, 1);   // /4
    c3a = make_c3(32, 32, 1);
    conv2 = make_convbn(32, 64, 3, 2, 1);   // /8   (stride 8)
    c3b = make_c3(64, 64, 1);               // -> C8
    conv3 = make_convbn(64, 128, 3, 2, 1);  // /16  (stride 16)
    c3c = make_c3(128, 128, 1);
    sppf = make_sppf(128, 128);             // -> C16
    c3h = make_c3(64 + 128, 64, 1);         // FPN head (cat C8 + up(C16))
    head_init(64, na * no, h8w, h8b);
    head_init(128, na * no, h16w, h16b);
}

std::vector<Tensor> SmallYolo::forward(const Tensor& x) {
    auto x0 = block(*this, conv0, x);           // /2
    auto x1 = block(*this, conv1, x0);          // /4
    x1 = fwd_c3(*this, c3a, x1);
    auto x2 = block(*this, conv2, x1);          // /8
    auto c8 = fwd_c3(*this, c3b, x2);           // stride-8 feature
    auto x3 = block(*this, conv3, c8);          // /16
    x3 = fwd_c3(*this, c3c, x3);
    auto c16 = fwd_sppf(*this, sppf, x3);       // stride-16 feature

    // FPN top-down
    auto up = upsample_nearest2d(c16, 2);
    auto p8 = fwd_c3(*this, c3h, cat_channels({c8, up}));

    auto pred8 = conv2d(p8, h8w, h8b, 1, 0);    // (N, na*no, G8, G8)
    auto pred16 = conv2d(c16, h16w, h16b, 1, 0); // (N, na*no, G16, G16)
    return {pred8, pred16};
}

std::vector<Tensor> SmallYolo::params() {
    std::vector<Tensor> p;
    collect(conv0, p); collect(conv1, p); collect(conv2, p); collect(conv3, p);
    collect_c3(c3a, p); collect_c3(c3b, p); collect_c3(c3c, p); collect_c3(c3h, p);
    collect(sppf.cv1, p); collect(sppf.cv2, p);
    p.push_back(h8w); p.push_back(h8b);
    p.push_back(h16w); p.push_back(h16b);
    return p;
}

// 3 classes distinguished by color (matches the LibTorch demo).
static const float kColor[3][3] = {{0.85f, 0.15f, 0.15f},
                                    {0.15f, 0.80f, 0.20f},
                                    {0.20f, 0.30f, 0.90f}};

void gen_batch(int N, int S, Tensor& imgs, std::vector<std::vector<Box>>& boxes) {
    imgs = Tensor::zeros({N, 3, S, S}, false);
    boxes.assign(N, {});
    auto& d = imgs.data();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < 3; ++c)
            for (int p = 0; p < S * S; ++p)
                d[(n * 3 + c) * S * S + p] = 0.35f + 0.12f * (randf() - 0.5f);

        int nobj = 1 + (randf() < 0.5f ? 1 : 0);  // 1 or 2 objects
        for (int k = 0; k < nobj; ++k) {
            int cls = int(randf() * 3) % 3;
            int bw = int(S * (0.2f + 0.25f * randf()));
            int bh = int(S * (0.2f + 0.25f * randf()));
            int x1 = int(randf() * (S - bw));
            int y1 = int(randf() * (S - bh));
            for (int c = 0; c < 3; ++c)
                for (int yy = y1; yy < y1 + bh; ++yy)
                    for (int xx = x1; xx < x1 + bw; ++xx)
                        d[(n * 3 + c) * S * S + yy * S + xx] =
                            kColor[cls][c] + 0.05f * (randf() - 0.5f);
            boxes[n].push_back({cls, (x1 + bw / 2.f) / S, (y1 + bh / 2.f) / S,
                                bw / (float)S, bh / (float)S});
        }
    }
}

// Differentiable CIoU between a predicted box (bx,by,bw,bh) and a constant
// target box (tx,ty,tw,th), all normalized, elementwise. Built entirely from
// the gradient-checked primitives.
static Tensor bbox_ciou(const Tensor& bx, const Tensor& by, const Tensor& bw,
                        const Tensor& bh, const Tensor& tx, const Tensor& ty,
                        const Tensor& tw, const Tensor& th) {
    const float eps = 1e-7f, pi = 3.14159265f;
    auto half = [](const Tensor& t) { return mul_scalar(t, 0.5f); };
    auto sq = [](const Tensor& t) { return mul(t, t); };
    auto px1 = sub(bx, half(bw)), px2 = add(bx, half(bw));
    auto py1 = sub(by, half(bh)), py2 = add(by, half(bh));
    auto tx1 = sub(tx, half(tw)), tx2 = add(tx, half(tw));
    auto ty1 = sub(ty, half(th)), ty2 = add(ty, half(th));

    auto iw = clamp_min(sub(minimum(px2, tx2), maximum(px1, tx1)), 0.f);
    auto ih = clamp_min(sub(minimum(py2, ty2), maximum(py1, ty1)), 0.f);
    auto inter = mul(iw, ih);
    auto uni = add_scalar(sub(add(mul(bw, bh), mul(tw, th)), inter), eps);
    auto iou = divide(inter, uni);

    auto cw = sub(maximum(px2, tx2), minimum(px1, tx1));
    auto ch = sub(maximum(py2, ty2), minimum(py1, ty1));
    auto c2 = add_scalar(add(sq(cw), sq(ch)), eps);
    auto rho2 = add(sq(sub(bx, tx)), sq(sub(by, ty)));

    auto at_t = atan_(divide(tw, add_scalar(th, eps)));
    auto at_p = atan_(divide(bw, add_scalar(bh, eps)));
    auto v = mul_scalar(sq(sub(at_t, at_p)), 4.f / (pi * pi));
    auto denom = add_scalar(add(add_scalar(mul_scalar(iou, -1.f), 1.f), v), eps);
    auto alpha = divide(v, denom);
    return sub(sub(iou, divide(rho2, c2)), mul(v, alpha));  // CIoU
}

// Build a constant (N,na,G,G) grid; filler(a) gives the per-anchor value,
// or use cell to fill gx/gy. mode: 0=gx, 1=gy, 2=anchor-w, 3=anchor-h.
static Tensor const_grid(int N, int na, int G, int mode, const std::vector<float>& anch) {
    auto t = Tensor::zeros({N, na, G, G}, false);
    for (int n = 0; n < N; ++n)
      for (int a = 0; a < na; ++a)
        for (int gy = 0; gy < G; ++gy)
          for (int gx = 0; gx < G; ++gx) {
            float v = 0;
            if (mode == 0) v = (float)gx;
            else if (mode == 1) v = (float)gy;
            else if (mode == 2) v = anch[2 * a];
            else v = anch[2 * a + 1];
            t.data()[((n * na + a) * G + gy) * G + gx] = v;
          }
    return t;
}

// Loss for one detection scale, anchor-based (na anchors) with CIoU box loss.
// Head channel layout is field-major: channel = field*na + a.
static Tensor loss_for_scale(SmallYolo& net, const Tensor& pred, int s,
                             const std::vector<std::vector<Box>>& boxes) {
    int N = pred.shape()[0], na = net.na, nc = net.nc, G = net.grids[s];
    const auto& anch = net.anchors[s];

    // target/mask grids: box fields & obj are (N,na,G,G); cls is (N,nc*na,G,G)
    auto tcx = Tensor::zeros({N, na, G, G}), tcy = Tensor::zeros({N, na, G, G});
    auto tw = Tensor::zeros({N, na, G, G}), th = Tensor::zeros({N, na, G, G});
    auto obj_t = Tensor::zeros({N, na, G, G});
    auto box_mask = Tensor::zeros({N, na, G, G});
    auto cls_t = Tensor::zeros({N, nc * na, G, G});
    auto cls_mask = Tensor::zeros({N, nc * na, G, G});

    int npos = 0;
    for (int n = 0; n < N; ++n)
        for (auto& b : boxes[n]) {
            int gx = std::min(G - 1, int(b.cx * G));
            int gy = std::min(G - 1, int(b.cy * G));
            // anchor matching: ratio test (< 4), fallback to best anchor
            int best_a = 0; float best_r = 1e9f; bool any = false;
            for (int a = 0; a < na; ++a) {
                float rw = std::max(b.w / anch[2 * a], anch[2 * a] / b.w);
                float rh = std::max(b.h / anch[2 * a + 1], anch[2 * a + 1] / b.h);
                float r = std::max(rw, rh);
                if (r < best_r) { best_r = r; best_a = a; }
                if (r < 4.0f) {
                    any = true;
                    int idx = ((n * na + a) * G + gy) * G + gx;
                    tcx.data()[idx] = b.cx; tcy.data()[idx] = b.cy;
                    tw.data()[idx] = b.w; th.data()[idx] = b.h;
                    obj_t.data()[idx] = 1.f; box_mask.data()[idx] = 1.f;
                    for (int c = 0; c < nc; ++c)
                        cls_mask.data()[((n * nc * na + (c * na + a)) * G + gy) * G + gx] = 1.f;
                    cls_t.data()[((n * nc * na + (b.cls * na + a)) * G + gy) * G + gx] = 1.f;
                    ++npos;
                }
            }
            if (!any) {  // fallback: force the best-matching anchor positive
                int a = best_a, idx = ((n * na + a) * G + gy) * G + gx;
                tcx.data()[idx] = b.cx; tcy.data()[idx] = b.cy;
                tw.data()[idx] = b.w; th.data()[idx] = b.h;
                obj_t.data()[idx] = 1.f; box_mask.data()[idx] = 1.f;
                for (int c = 0; c < nc; ++c)
                    cls_mask.data()[((n * nc * na + (c * na + a)) * G + gy) * G + gx] = 1.f;
                cls_t.data()[((n * nc * na + (b.cls * na + a)) * G + gy) * G + gx] = 1.f;
                ++npos;
            }
        }
    float inv_pos = 1.0f / std::max(1, npos);
    auto neg_mask = Tensor::zeros({N, na, G, G});
    for (int i = 0; i < neg_mask.numel(); ++i) neg_mask.data()[i] = 1.f - box_mask.data()[i];
    float inv_neg = 1.0f / std::max(1, N * na * G * G - npos);

    // constant grids for decode
    auto gx_g = const_grid(N, na, G, 0, anch), gy_g = const_grid(N, na, G, 1, anch);
    auto aw_g = const_grid(N, na, G, 2, anch), ah_g = const_grid(N, na, G, 3, anch);

    // decode (v5 formula), fields sliced per-anchor-block (field-major layout)
    auto slc = [&](int f) { return slice_channels(pred, f * na, (f + 1) * na); };
    auto sx = sigmoid(slc(0)), sy = sigmoid(slc(1));
    auto sw = sigmoid(slc(2)), sh = sigmoid(slc(3));
    auto objp = sigmoid(slc(4));
    auto clsp = sigmoid(slice_channels(pred, 5 * na, net.no * na));  // (N,nc*na,G,G)

    float invG = 1.f / G;
    auto bx = mul_scalar(add(add_scalar(mul_scalar(sx, 2.f), -0.5f), gx_g), invG);
    auto by = mul_scalar(add(add_scalar(mul_scalar(sy, 2.f), -0.5f), gy_g), invG);
    auto bw = mul(mul(mul_scalar(sw, 2.f), mul_scalar(sw, 2.f)), aw_g);  // (2sw)^2 * aw
    auto bh = mul(mul(mul_scalar(sh, 2.f), mul_scalar(sh, 2.f)), ah_g);

    auto ciou = bbox_ciou(bx, by, bw, bh, tcx, tcy, tw, th);
    auto Lbox = mul_scalar(sum(mul(box_mask, add_scalar(mul_scalar(ciou, -1.f), 1.f))), inv_pos);

    // Objectness target = detached CIoU at positives (YOLOv5-style). Confidence
    // then reflects box quality, so background / low-IoU boxes stay low -> far
    // fewer false positives. (ciou.data() holds the eagerly-computed forward
    // values; obj_t is a constant leaf, so this is effectively a stop-gradient.)
    for (int i = 0; i < obj_t.numel(); ++i)
        if (box_mask.data()[i] > 0.5f) obj_t.data()[i] = std::max(0.f, ciou.data()[i]);

    auto sq = [](const Tensor& d) { return mul(d, d); };
    auto obj_err = sq(sub(objp, obj_t));
    // negatives weighted 2x to suppress background objectness (fewer false positives)
    auto Lobj = add(mul_scalar(sum(mul(box_mask, obj_err)), inv_pos),
                    mul_scalar(sum(mul(neg_mask, obj_err)), inv_neg * 2.0f));
    auto Lcls = mul_scalar(sum(mul(cls_mask, sq(sub(clsp, cls_t)))), inv_pos / nc);

    // box gain 5, class gain 2 (stronger class supervision -> less person/car mixup)
    return add(add(mul_scalar(Lbox, 5.0f), Lobj), mul_scalar(Lcls, 2.0f));
}

// Route each GT to ONE scale by size: small -> fine (stride 8), large -> coarse.
static int assign_scale(const Box& b, int nscales) {
    float sz = std::max(b.w, b.h);
    return (nscales > 1 && sz > 0.33f) ? 1 : 0;
}

Tensor compute_loss(SmallYolo& net, const std::vector<Tensor>& preds,
                    const std::vector<std::vector<Box>>& boxes) {
    int nscales = (int)preds.size();
    Tensor total;
    for (int s = 0; s < nscales; ++s) {
        std::vector<std::vector<Box>> sub(boxes.size());
        for (size_t n = 0; n < boxes.size(); ++n)
            for (auto& b : boxes[n])
                if (assign_scale(b, nscales) == s) sub[n].push_back(b);
        auto l = loss_for_scale(net, preds[s], s, sub);
        total = (s == 0) ? l : add(total, l);
    }
    return total;
}

std::vector<Box> decode(SmallYolo& net, const std::vector<Tensor>& preds, int n,
                        float conf_thres, std::vector<float>* confs) {
    int nc = net.nc, na = net.na;
    auto sig = [](float v) { return 1.f / (1.f + std::exp(-v)); };
    std::vector<Box> out;
    for (size_t s = 0; s < preds.size(); ++s) {
        int G = net.grids[s];
        const auto& anch = net.anchors[s];
        auto& d = preds[s].data();
        // channel = field*na + a
        auto at = [&](int field, int a, int gy, int gx) {
            int ch = field * na + a;
            return d[((n * (net.no * na) + ch) * G + gy) * G + gx];
        };
        for (int a = 0; a < na; ++a)
            for (int gy = 0; gy < G; ++gy)
                for (int gx = 0; gx < G; ++gx) {
                    float obj = sig(at(4, a, gy, gx));
                    if (obj < conf_thres) continue;
                    int best = 0; float bv = -1e9f;
                    for (int c = 0; c < nc; ++c) {
                        float v = sig(at(5 + c, a, gy, gx));
                        if (v > bv) { bv = v; best = c; }
                    }
                    float sw = sig(at(2, a, gy, gx)), sh = sig(at(3, a, gy, gx));
                    float bw = (2 * sw) * (2 * sw) * anch[2 * a];
                    float bh = (2 * sh) * (2 * sh) * anch[2 * a + 1];
                    if (bw * bh < 1e-4f) continue;
                    out.push_back({best,
                                   (gx + 2 * sig(at(0, a, gy, gx)) - 0.5f) / G,
                                   (gy + 2 * sig(at(1, a, gy, gx)) - 0.5f) / G, bw, bh});
                    if (confs) confs->push_back(obj);
                }
    }
    return out;
}

// Greedy per-class NMS over normalized cx,cy,w,h boxes.
static std::vector<int> nms(const std::vector<Box>& b, const std::vector<float>& conf,
                            float iou_thr) {
    auto iou = [](const Box& p, const Box& q) {
        float ax1 = p.cx - p.w / 2, ay1 = p.cy - p.h / 2, ax2 = p.cx + p.w / 2, ay2 = p.cy + p.h / 2;
        float bx1 = q.cx - q.w / 2, by1 = q.cy - q.h / 2, bx2 = q.cx + q.w / 2, by2 = q.cy + q.h / 2;
        float iw = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
        float ih = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
        float inter = iw * ih, uni = p.w * p.h + q.w * q.h - inter;
        return uni > 0 ? inter / uni : 0.f;
    };
    std::vector<int> order(b.size());
    for (size_t i = 0; i < b.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int i, int j) { return conf[i] > conf[j]; });
    std::vector<int> keep;
    std::vector<bool> dead(b.size(), false);
    for (int idx : order) {
        if (dead[idx]) continue;
        keep.push_back(idx);
        for (int o : order)
            if (!dead[o] && o != idx && b[o].cls == b[idx].cls && iou(b[idx], b[o]) > iou_thr)
                dead[o] = true;
    }
    return keep;
}

// Draw a rectangle outline into a (3,S,S) float image (normalized box).
static void draw_box(std::vector<float>& img, int S, const Box& b,
                     float r, float g, float bl, int th = 2) {
    int x1 = int((b.cx - b.w / 2) * S), y1 = int((b.cy - b.h / 2) * S);
    int x2 = int((b.cx + b.w / 2) * S), y2 = int((b.cy + b.h / 2) * S);
    auto cx = [&](int v) { return std::max(0, std::min(S - 1, v)); };
    x1 = cx(x1); x2 = cx(x2); y1 = cx(y1); y2 = cx(y2);
    float col[3] = {r, g, bl};
    auto set = [&](int x, int y) { for (int c = 0; c < 3; ++c) img[(c * S + y) * S + x] = col[c]; };
    for (int t = 0; t < th; ++t) {
        for (int x = x1; x <= x2; ++x) { set(x, cx(y1 + t)); set(x, cx(y2 - t)); }
        for (int y = y1; y <= y2; ++y) { set(cx(x1 + t), y); set(cx(x2 - t), y); }
    }
}

// Write a (3,S,S) float [0,1] image as PNG via stb_image_write.
static void write_png(const std::string& path, const std::vector<float>& img, int S) {
    std::vector<unsigned char> rgb((size_t)S * S * 3);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = img[(c * S + y) * S + x];
                rgb[(y * S + x) * 3 + c] =
                    (unsigned char)std::max(0.f, std::min(255.f, v * 255));
            }
    stbi_write_png(path.c_str(), S, S, 3, rgb.data(), S * 3);
}

int train_demo(const std::string& data_dir, int nc, int input_size) {
    seed(7);
    SmallYolo net(nc, input_size);
    auto params = net.params();
    // momentum buffers
    std::vector<std::vector<float>> vel(params.size());
    for (size_t i = 0; i < params.size(); ++i) vel[i].assign(params[i].numel(), 0.f);

    // deeper BN net -> lower base lr + warmup + cosine decay, for stability
    float lr0 = 0.03f, mom = 0.9f;
    int N = 12, S = net.S, steps = 2500, warmup = 50;

    // load shared on-disk dataset if a directory was given
    bool disk = !data_dir.empty();
    Dataset ds;
    if (disk) {
        ds = load_disk_dataset(data_dir, S);
        if (ds.size() == 0) { printf("no .ppm images under %s\n", data_dir.c_str()); return 1; }
    }

    printf("== train from-scratch YOLO (self-made autograd, zero deps) ==\n");
    printf("   input %dx%d, grids %dx%d + %dx%d (FPN, BatchNorm), %d classes, batch %d\n",
           S, S, net.grids[0], net.grids[0], net.grids[1], net.grids[1], net.nc, N);
    printf("   data: %s\n", disk ? (data_dir + " (disk, " + std::to_string(ds.size()) + " imgs)").c_str()
                                  : "in-memory synthetic");

    net.training = true;
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; ++step) {
        Tensor imgs; std::vector<std::vector<Box>> boxes;
        if (disk) batch_from(ds, N, imgs, boxes, true);
        else gen_batch(N, S, imgs, boxes);
        // linear warmup then cosine decay (warms down to 5% of lr0)
        float lr;
        if (step < warmup) {
            lr = lr0 * (float)step / warmup;
        } else {
            float p = (float)(step - warmup) / (steps - warmup);
            lr = lr0 * (0.05f + 0.95f * 0.5f * (1 + std::cos(3.14159265f * p)));
        }
        for (auto& p : params) p.zero_grad();
        auto pred = net.forward(imgs);
        auto loss = compute_loss(net, pred, boxes);
        loss.backward();
        for (size_t i = 0; i < params.size(); ++i) {
            auto& pd = params[i].data(); auto& pg = params[i].grad();
            for (size_t j = 0; j < pd.size(); ++j) {
                vel[i][j] = mom * vel[i][j] - lr * pg[j];
                pd[j] += vel[i][j];
            }
        }
        if (step % 50 == 0) {
            printf("  step %3d  loss %.4f  lr %.4f\n", step, loss.item(), lr);
            std::fflush(stdout);  // show live progress even when redirected to a file
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("\ntrained %d steps in %.1fs  (%.3fs/step)\n", steps, secs, secs / (steps + 1));

    // --- evaluate on fresh images (BN uses running stats now) ---
    printf("== evaluate (obj>0.6) ==\n");
    net.training = false;
    Tensor imgs; std::vector<std::vector<Box>> boxes;
    if (disk) batch_from(ds, 4, imgs, boxes, false);
    else gen_batch(4, S, imgs, boxes);
    auto pred = net.forward(imgs);
    // class names: rgb demo for nc==3, person/bus for nc==2, else generic
    std::vector<std::string> names;
    if (nc == 3) names = {"red", "green", "blue"};
    else if (nc == 2) names = {"person", "car"};
    else for (int c = 0; c < nc; ++c) names.push_back("cls" + std::to_string(c));
    auto nm = [&](int c) { return (c >= 0 && c < nc) ? names[c].c_str() : "?"; };

    int neval = std::min<int>(4, (int)(disk ? ds.size() : 4));
    int hits = 0, total = 0;
    for (int n = 0; n < neval; ++n) {
        printf(" image %d:\n", n);
        // copy this image out of the batch tensor for drawing
        std::vector<float> canvas(imgs.data().begin() + (size_t)n * 3 * S * S,
                                  imgs.data().begin() + (size_t)(n + 1) * 3 * S * S);
        for (auto& g : boxes[n]) {
            printf("   GT  %-6s cx=%.2f cy=%.2f w=%.2f h=%.2f\n",
                   nm(g.cls), g.cx, g.cy, g.w, g.h);
            draw_box(canvas, S, g, 1.f, 0.f, 0.f, 1);   // GT = thin red
            ++total;
        }
        std::vector<float> confs;
        auto dets = decode(net, pred, n, 0.7f, &confs);
        auto keep = nms(dets, confs, 0.3f);   // dedupe neighbouring cells
        for (int k : keep) {
            auto& dbox = dets[k];
            printf("   DET %-6s cx=%.2f cy=%.2f w=%.2f h=%.2f  conf=%.2f\n",
                   nm(dbox.cls), dbox.cx, dbox.cy, dbox.w, dbox.h, confs[k]);
            draw_box(canvas, S, dbox, 0.f, 1.f, 0.f, 2);  // DET = thick green
            for (auto& g : boxes[n])
                if (g.cls == dbox.cls && std::fabs(g.cx - dbox.cx) < 0.15f &&
                    std::fabs(g.cy - dbox.cy) < 0.15f) { ++hits; break; }
        }
        std::string vp = "viz_" + std::to_string(n) + ".png";
        write_png(vp, canvas, S);
        printf("   -> wrote %s (GT=red, DET=green)\n", vp.c_str());
    }
    printf("\n matched %d detections to GT (of %d GT objects)\n", hits, total);
    return 0;
}

}  // namespace scratch
