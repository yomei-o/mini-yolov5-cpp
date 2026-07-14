#include "autograd.h"
#include <cmath>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <mutex>

namespace ag {

// ---- deterministic RNG (xorshift128+) so runs are reproducible ------------
static uint64_t g_s0 = 0x9E3779B97F4A7C15ull, g_s1 = 0xBF58476D1CE4E5B9ull;
void seed(uint64_t s) {
    g_s0 = s ? s : 1;
    g_s1 = s ^ 0xD1B54A32D192ED03ull;
}
static uint64_t next_u64() {
    uint64_t x = g_s0, y = g_s1;
    g_s0 = y;
    x ^= x << 23;
    g_s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return g_s1 + y;
}
float randf() { return (next_u64() >> 11) * (1.0f / 9007199254740992.0f); }
static float randn_std() {  // Box–Muller
    float u1 = randf(), u2 = randf();
    if (u1 < 1e-12f) u1 = 1e-12f;
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
}

// ---- Tensor factories ------------------------------------------------------
Tensor Tensor::zeros(std::vector<int> shape, bool rg) {
    auto n = std::make_shared<Node>();
    n->shape = std::move(shape);
    n->data.assign(n->numel(), 0.f);
    n->grad.assign(n->numel(), 0.f);
    n->requires_grad = rg;
    n->op = "leaf";
    return Tensor(n);
}
Tensor Tensor::from(std::vector<float> data, std::vector<int> shape, bool rg) {
    auto t = zeros(std::move(shape), rg);
    t.n->data = std::move(data);
    return t;
}
Tensor Tensor::randn(std::vector<int> shape, float std, bool rg) {
    auto t = zeros(std::move(shape), rg);
    for (auto& v : t.n->data) v = randn_std() * std;
    return t;
}

void Tensor::zero_grad() const { std::fill(n->grad.begin(), n->grad.end(), 0.f); }

void Tensor::backward() const {
    // reverse topological order over the graph rooted at this node
    std::vector<Node*> topo;
    std::unordered_set<Node*> seen;
    std::function<void(Node*)> build = [&](Node* v) {
        if (seen.count(v)) return;
        seen.insert(v);
        for (auto& p : v->parents) build(p.get());
        topo.push_back(v);
    };
    build(n.get());

    n->grad.assign(n->numel(), 0.f);
    n->grad[0] = 1.0f;  // seed dLoss/dLoss = 1 (expects a scalar root)
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn();
}

Tensor make_op(std::vector<int> shape, std::vector<NodePtr> parents,
               const std::string& op) {
    auto o = std::make_shared<Node>();
    o->shape = std::move(shape);
    o->data.assign(o->numel(), 0.f);
    o->grad.assign(o->numel(), 0.f);
    o->parents = std::move(parents);
    o->op = op;
    for (auto& p : o->parents) o->requires_grad |= p->requires_grad;
    return Tensor(o);
}

// ---- elementwise -----------------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b) {
    auto out = make_op(a.shape(), {a.n, b.n}, "add");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = a.data()[i] + b.data()[i];
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            A->grad[i] += O->grad[i];
            B->grad[i] += O->grad[i];
        }
    };
    return out;
}
Tensor sub(const Tensor& a, const Tensor& b) {
    auto out = make_op(a.shape(), {a.n, b.n}, "sub");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = a.data()[i] - b.data()[i];
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            A->grad[i] += O->grad[i];
            B->grad[i] -= O->grad[i];
        }
    };
    return out;
}
Tensor mul(const Tensor& a, const Tensor& b) {
    auto out = make_op(a.shape(), {a.n, b.n}, "mul");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = a.data()[i] * b.data()[i];
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            A->grad[i] += B->data[i] * O->grad[i];
            B->grad[i] += A->data[i] * O->grad[i];
        }
    };
    return out;
}
Tensor mul_scalar(const Tensor& a, float s) {
    auto out = make_op(a.shape(), {a.n}, "mul_scalar");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = a.data()[i] * s;
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O, s]() {
        for (size_t i = 0; i < O->grad.size(); ++i) A->grad[i] += s * O->grad[i];
    };
    return out;
}
Tensor add_scalar(const Tensor& a, float s) {
    auto out = make_op(a.shape(), {a.n}, "add_scalar");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = a.data()[i] + s;
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) A->grad[i] += O->grad[i];
    };
    return out;
}
Tensor sum(const Tensor& a) {
    auto out = make_op({1}, {a.n}, "sum");
    float s = 0;
    for (int i = 0; i < a.numel(); ++i) s += a.data()[i];
    out.data()[0] = s;
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < A->grad.size(); ++i) A->grad[i] += O->grad[0];
    };
    return out;
}
Tensor mean(const Tensor& a) {
    int n = a.numel();
    return mul_scalar(sum(a), 1.0f / n);
}

// ---- matmul ----------------------------------------------------------------
Tensor matmul(const Tensor& a, const Tensor& b) {
    int m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    auto out = make_op({m, n}, {a.n, b.n}, "matmul");
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0;
            for (int p = 0; p < k; ++p) s += a.data()[i * k + p] * b.data()[p * n + j];
            out.data()[i * n + j] = s;
        }
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O, m, k, n]() {
        // dA = dO @ B^T ; dB = A^T @ dO
        for (int i = 0; i < m; ++i)
            for (int p = 0; p < k; ++p) {
                float s = 0;
                for (int j = 0; j < n; ++j) s += O->grad[i * n + j] * B->data[p * n + j];
                A->grad[i * k + p] += s;
            }
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j) {
                float s = 0;
                for (int i = 0; i < m; ++i) s += A->data[i * k + p] * O->grad[i * n + j];
                B->grad[p * n + j] += s;
            }
    };
    return out;
}

// ---- conv / pooling (NCHW) -------------------------------------------------
// Cache-friendly GEMM: C(m,n) = A(m,k) * B(k,n). The i-p-j loop order keeps the
// inner loop contiguous in B and C, which is ~an order of magnitude faster than
// the naive i-j-p order and lets the compiler auto-vectorize it.
static void gemm(const float* A, const float* B, float* C, int m, int k, int n,
                 bool accum) {
    if (!accum) std::fill(C, C + (size_t)m * n, 0.f);
    for (int i = 0; i < m; ++i) {
        const float* Ai = A + (size_t)i * k;
        float* Ci = C + (size_t)i * n;
        for (int p = 0; p < k; ++p) {
            float a = Ai[p];
            const float* Bp = B + (size_t)p * n;
            for (int j = 0; j < n; ++j) Ci[j] += a * Bp[j];
        }
    }
}

// im2col: unfold one image's receptive fields into col(K, P),
// K = C*kh*kw, P = OH*OW. Then conv == W(O,K) @ col(K,P).
static void im2col(const float* x, int C, int H, int W, int kh, int kw,
                   int stride, int pad, int OH, int OW, float* col) {
    int P = OH * OW;
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < kh; ++i)
        for (int j = 0; j < kw; ++j) {
            float* crow = col + (size_t)((c * kh + i) * kw + j) * P;
            for (int oh = 0; oh < OH; ++oh) {
                int ih = oh * stride + i - pad;
                for (int ow = 0; ow < OW; ++ow) {
                    int iw = ow * stride + j - pad;
                    crow[oh * OW + ow] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                             ? x[(c * H + ih) * W + iw] : 0.f;
                }
            }
        }
}

// col2im: scatter-add columns back into an image gradient (transpose of im2col).
static void col2im(const float* col, int C, int H, int W, int kh, int kw,
                   int stride, int pad, int OH, int OW, float* dx) {
    int P = OH * OW;
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < kh; ++i)
        for (int j = 0; j < kw; ++j) {
            const float* crow = col + (size_t)((c * kh + i) * kw + j) * P;
            for (int oh = 0; oh < OH; ++oh) {
                int ih = oh * stride + i - pad;
                if (ih < 0 || ih >= H) continue;
                for (int ow = 0; ow < OW; ++ow) {
                    int iw = ow * stride + j - pad;
                    if (iw < 0 || iw >= W) continue;
                    dx[(c * H + ih) * W + iw] += crow[oh * OW + ow];
                }
            }
        }
}

// Run body(i) for i in [0,n) across std::thread workers (standard library only).
// body(i) must touch only data private to index i (plus its own locals).
static void parallel_for(int n, const std::function<void(int)>& body) {
    unsigned hw = std::thread::hardware_concurrency();
    int T = std::max(1, std::min((int)(hw ? hw : 1), n));
    if (T == 1) { for (int i = 0; i < n; ++i) body(i); return; }
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t)
        ths.emplace_back([&, t] { for (int i = t; i < n; i += T) body(i); });
    for (auto& th : ths) th.join();
}

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride, int pad) {
    int N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int O = w.shape()[0], kh = w.shape()[2], kw = w.shape()[3];
    int OH = (H + 2 * pad - kh) / stride + 1;
    int OW = (W + 2 * pad - kw) / stride + 1;
    int K = C * kh * kw, P = OH * OW;
    auto out = make_op({N, O, OH, OW}, {x.n, w.n, b.n}, "conv2d");

    auto& xd = x.data(); auto& wd = w.data(); auto& bd = b.data(); auto& od = out.data();
    // Forward: images are independent -> parallelize over the batch. Each worker
    // uses its own im2col buffer and writes only its output slice (no locking).
    parallel_for(N, [&](int n) {
        std::vector<float> col((size_t)K * P);
        im2col(xd.data() + (size_t)n * C * H * W, C, H, W, kh, kw, stride, pad, OH, OW, col.data());
        float* on = od.data() + (size_t)n * O * P;
        gemm(wd.data(), col.data(), on, O, K, P, /*accum=*/false);  // (O,K)*(K,P)
        for (int o = 0; o < O; ++o) {
            float bo = bd[o], *orow = on + (size_t)o * P;
            for (int p = 0; p < P; ++p) orow[p] += bo;
        }
    });

    Node *X = x.n.get(), *Wt = w.n.get(), *B = b.n.get(), *Ot = out.n.get();
    out.n->backward_fn = [=]() {
        unsigned hw = std::thread::hardware_concurrency();
        int T = std::max(1, std::min((int)(hw ? hw : 1), N));
        std::mutex mtx;
        // dInput is written per-image (disjoint slices); dW/dBias are accumulated
        // into thread-local buffers and reduced under a lock at the end.
        auto worker = [&](int t) {
            std::vector<float> col((size_t)K * P), dcol((size_t)K * P);
            std::vector<float> dW((size_t)O * K, 0.f), dB(O, 0.f);
            for (int n = t; n < N; n += T) {
                const float* gn = Ot->grad.data() + (size_t)n * O * P;  // dOut (O,P)
                for (int o = 0; o < O; ++o) {
                    const float* gr = gn + (size_t)o * P;
                    float s = 0; for (int p = 0; p < P; ++p) s += gr[p];
                    dB[o] += s;
                }
                im2col(X->data.data() + (size_t)n * C * H * W, C, H, W, kh, kw, stride, pad, OH, OW, col.data());
                // dW(O,K) += dOut(O,P) . col(K,P)^T
                for (int o = 0; o < O; ++o) {
                    const float* gr = gn + (size_t)o * P;
                    float* dwo = dW.data() + (size_t)o * K;
                    for (int r = 0; r < K; ++r) {
                        const float* cr = col.data() + (size_t)r * P;
                        float s = 0; for (int p = 0; p < P; ++p) s += gr[p] * cr[p];
                        dwo[r] += s;
                    }
                }
                // dcol(K,P) = W(O,K)^T . dOut(O,P)
                std::fill(dcol.begin(), dcol.end(), 0.f);
                for (int o = 0; o < O; ++o) {
                    const float* wo = Wt->data.data() + (size_t)o * K;
                    const float* gr = gn + (size_t)o * P;
                    for (int r = 0; r < K; ++r) {
                        float wv = wo[r]; float* dr = dcol.data() + (size_t)r * P;
                        for (int p = 0; p < P; ++p) dr[p] += wv * gr[p];
                    }
                }
                col2im(dcol.data(), C, H, W, kh, kw, stride, pad, OH, OW,
                       X->grad.data() + (size_t)n * C * H * W);  // disjoint per n
            }
            std::lock_guard<std::mutex> lk(mtx);
            for (size_t i = 0; i < dW.size(); ++i) Wt->grad[i] += dW[i];
            for (int o = 0; o < O; ++o) B->grad[o] += dB[o];
        };
        if (T == 1) { worker(0); return; }
        std::vector<std::thread> ths;
        for (int t = 0; t < T; ++t) ths.emplace_back(worker, t);
        for (auto& th : ths) th.join();
    };
    return out;
}

Tensor maxpool2d(const Tensor& x, int k, int stride, int pad) {
    int N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int OH = (H + 2 * pad - k) / stride + 1;
    int OW = (W + 2 * pad - k) / stride + 1;
    auto out = make_op({N, C, OH, OW}, {x.n}, "maxpool2d");
    // remember argmax index (into x) for each output cell to route gradients
    auto argmax = std::make_shared<std::vector<int>>(out.numel(), -1);

    auto& xd = x.data(); auto& od = out.data();
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int oh = 0; oh < OH; ++oh)
          for (int ow = 0; ow < OW; ++ow) {
            float best = -1e30f; int bi = -1;
            for (int i = 0; i < k; ++i)
              for (int j = 0; j < k; ++j) {
                  int ih = oh * stride + i - pad, iw = ow * stride + j - pad;
                  if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                  int xi = ((n * C + c) * H + ih) * W + iw;
                  if (xd[xi] > best) { best = xd[xi]; bi = xi; }
              }
            int oi = ((n * C + c) * OH + oh) * OW + ow;
            od[oi] = best; (*argmax)[oi] = bi;
          }

    Node *X = x.n.get(), *Ot = out.n.get();
    out.n->backward_fn = [=]() {
        for (size_t oi = 0; oi < Ot->grad.size(); ++oi) {
            int bi = (*argmax)[oi];
            if (bi >= 0) X->grad[bi] += Ot->grad[oi];
        }
    };
    return out;
}

Tensor upsample_nearest2d(const Tensor& x, int scale) {
    int N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int OH = H * scale, OW = W * scale;
    auto out = make_op({N, C, OH, OW}, {x.n}, "upsample");
    auto& xd = x.data(); auto& od = out.data();
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int oh = 0; oh < OH; ++oh)
          for (int ow = 0; ow < OW; ++ow)
            od[((n * C + c) * OH + oh) * OW + ow] =
                xd[((n * C + c) * H + oh / scale) * W + ow / scale];
    Node *X = x.n.get(), *Ot = out.n.get();
    out.n->backward_fn = [=]() {
        for (int n = 0; n < N; ++n)
          for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < OH; ++oh)
              for (int ow = 0; ow < OW; ++ow)
                X->grad[((n * C + c) * H + oh / scale) * W + ow / scale] +=
                    Ot->grad[((n * C + c) * OH + oh) * OW + ow];
    };
    return out;
}

Tensor batchnorm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   Tensor& running_mean, Tensor& running_var,
                   bool training, float momentum, float eps) {
    int N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int HW = H * W, M = N * HW;
    auto out = make_op(x.shape(), {x.n, gamma.n, beta.n}, "batchnorm");
    auto& xd = x.data(); auto& od = out.data();
    auto& g = gamma.data(); auto& b = beta.data();

    std::vector<float> mu(C), inv(C);   // per-channel stats used in forward+backward
    if (training) {
        for (int c = 0; c < C; ++c) {
            double s = 0;
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) s += xd[(n * C + c) * HW + i];
            float m = float(s / M);
            double v = 0;
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) {
                    float d = xd[(n * C + c) * HW + i] - m;
                    v += double(d) * d;
                }
            float var = float(v / M);
            mu[c] = m; inv[c] = 1.f / std::sqrt(var + eps);
            // update running stats (side effect, no grad)
            running_mean.data()[c] = (1 - momentum) * running_mean.data()[c] + momentum * m;
            running_var.data()[c] = (1 - momentum) * running_var.data()[c] + momentum * var;
        }
    } else {
        for (int c = 0; c < C; ++c) {
            mu[c] = running_mean.data()[c];
            inv[c] = 1.f / std::sqrt(running_var.data()[c] + eps);
        }
    }
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < HW; ++i) {
                int idx = (n * C + c) * HW + i;
                od[idx] = g[c] * (xd[idx] - mu[c]) * inv[c] + b[c];
            }

    Node *X = x.n.get(), *G = gamma.n.get(), *B = beta.n.get(), *O = out.n.get();
    out.n->backward_fn = [=]() {
        // standard BatchNorm backward (per channel over M elements)
        for (int c = 0; c < C; ++c) {
            float sum_dy = 0, sum_dy_xhat = 0;
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) {
                    int idx = (n * C + c) * HW + i;
                    float dy = O->grad[idx];
                    float xhat = (X->data[idx] - mu[c]) * inv[c];
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                }
            B->grad[c] += sum_dy;
            G->grad[c] += sum_dy_xhat;
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) {
                    int idx = (n * C + c) * HW + i;
                    float dy = O->grad[idx];
                    float xhat = (X->data[idx] - mu[c]) * inv[c];
                    float dxhat = dy * g[c];
                    // dx = inv/M * (M*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
                    // note sum(dxhat)=g*sum_dy, sum(dxhat*xhat)=g*sum_dy_xhat
                    X->grad[idx] += inv[c] / M *
                        (M * dxhat - g[c] * sum_dy - xhat * g[c] * sum_dy_xhat);
                }
        }
    };
    return out;
}

Tensor add_bias_nchw(const Tensor& x, const Tensor& b) {
    int N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    auto out = make_op(x.shape(), {x.n, b.n}, "add_bias");
    auto& xd = x.data(); auto& bd = b.data(); auto& od = out.data();
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int hw = 0; hw < H * W; ++hw)
          od[(n * C + c) * H * W + hw] = xd[(n * C + c) * H * W + hw] + bd[c];
    Node *X = x.n.get(), *B = b.n.get(), *Ot = out.n.get();
    out.n->backward_fn = [=]() {
        for (int n = 0; n < N; ++n)
          for (int c = 0; c < C; ++c)
            for (int hw = 0; hw < H * W; ++hw) {
                float g = Ot->grad[(n * C + c) * H * W + hw];
                X->grad[(n * C + c) * H * W + hw] += g;
                B->grad[c] += g;
            }
    };
    return out;
}

Tensor cat_channels(const std::vector<Tensor>& xs) {
    int N = xs[0].shape()[0], H = xs[0].shape()[2], W = xs[0].shape()[3];
    int Ctot = 0;
    std::vector<NodePtr> parents;
    for (auto& t : xs) { Ctot += t.shape()[1]; parents.push_back(t.n); }
    auto out = make_op({N, Ctot, H, W}, parents, "cat");
    auto& od = out.data();
    for (int n = 0; n < N; ++n) {
        int coff = 0;
        for (auto& t : xs) {
            int C = t.shape()[1];
            for (int c = 0; c < C; ++c)
              for (int hw = 0; hw < H * W; ++hw)
                od[((n * Ctot) + (coff + c)) * H * W + hw] =
                    t.data()[(n * C + c) * H * W + hw];
            coff += C;
        }
    }
    std::vector<Node*> src; for (auto& t : xs) src.push_back(t.n.get());
    std::vector<int> chans; for (auto& t : xs) chans.push_back(t.shape()[1]);
    Node* Ot = out.n.get();
    out.n->backward_fn = [=]() {
        for (int n = 0; n < N; ++n) {
            int coff = 0;
            for (size_t s = 0; s < src.size(); ++s) {
                int C = chans[s];
                for (int c = 0; c < C; ++c)
                  for (int hw = 0; hw < H * W; ++hw)
                    src[s]->grad[(n * C + c) * H * W + hw] +=
                        Ot->grad[((n * Ctot) + (coff + c)) * H * W + hw];
                coff += C;
            }
        }
    };
    return out;
}

Tensor slice_channels(const Tensor& x, int c0, int c1) {
    int N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int Cs = c1 - c0;
    auto out = make_op({N, Cs, H, W}, {x.n}, "slice");
    auto& xd = x.data(); auto& od = out.data();
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < Cs; ++c)
        for (int hw = 0; hw < H * W; ++hw)
          od[(n * Cs + c) * H * W + hw] = xd[(n * C + (c0 + c)) * H * W + hw];
    Node *X = x.n.get(), *Ot = out.n.get();
    out.n->backward_fn = [=]() {
        for (int n = 0; n < N; ++n)
          for (int c = 0; c < Cs; ++c)
            for (int hw = 0; hw < H * W; ++hw)
              X->grad[(n * C + (c0 + c)) * H * W + hw] +=
                  Ot->grad[(n * Cs + c) * H * W + hw];
    };
    return out;
}

// ---- elementwise math (used to build CIoU from differentiable pieces) ------
Tensor maximum(const Tensor& a, const Tensor& b) {
    auto out = make_op(a.shape(), {a.n, b.n}, "maximum");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = std::max(a.data()[i], b.data()[i]);
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i)
            (A->data[i] >= B->data[i] ? A->grad[i] : B->grad[i]) += O->grad[i];
    };
    return out;
}
Tensor minimum(const Tensor& a, const Tensor& b) {
    auto out = make_op(a.shape(), {a.n, b.n}, "minimum");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = std::min(a.data()[i], b.data()[i]);
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i)
            (A->data[i] <= B->data[i] ? A->grad[i] : B->grad[i]) += O->grad[i];
    };
    return out;
}
Tensor divide(const Tensor& a, const Tensor& b) {
    auto out = make_op(a.shape(), {a.n, b.n}, "divide");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = a.data()[i] / b.data()[i];
    Node *A = a.n.get(), *B = b.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, B, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            float bv = B->data[i];
            A->grad[i] += O->grad[i] / bv;
            B->grad[i] += -O->grad[i] * A->data[i] / (bv * bv);
        }
    };
    return out;
}
Tensor sqrt_(const Tensor& a) {
    auto out = make_op(a.shape(), {a.n}, "sqrt");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = std::sqrt(a.data()[i]);
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i)
            A->grad[i] += O->grad[i] * 0.5f / (O->data[i] + 1e-12f);
    };
    return out;
}
Tensor atan_(const Tensor& a) {
    auto out = make_op(a.shape(), {a.n}, "atan");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = std::atan(a.data()[i]);
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            float x = A->data[i];
            A->grad[i] += O->grad[i] / (1 + x * x);
        }
    };
    return out;
}
Tensor clamp_min(const Tensor& a, float m) {
    auto out = make_op(a.shape(), {a.n}, "clamp_min");
    for (int i = 0; i < out.numel(); ++i) out.data()[i] = std::max(a.data()[i], m);
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O, m]() {
        for (size_t i = 0; i < O->grad.size(); ++i)
            if (A->data[i] > m) A->grad[i] += O->grad[i];
    };
    return out;
}

// ---- activations -----------------------------------------------------------
Tensor relu(const Tensor& a) {
    auto out = make_op(a.shape(), {a.n}, "relu");
    for (int i = 0; i < a.numel(); ++i) out.data()[i] = a.data()[i] > 0 ? a.data()[i] : 0;
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i)
            A->grad[i] += (O->data[i] > 0 ? 1.f : 0.f) * O->grad[i];
    };
    return out;
}
Tensor sigmoid(const Tensor& a) {
    auto out = make_op(a.shape(), {a.n}, "sigmoid");
    for (int i = 0; i < a.numel(); ++i) out.data()[i] = 1.f / (1.f + std::exp(-a.data()[i]));
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            float s = O->data[i];
            A->grad[i] += s * (1 - s) * O->grad[i];
        }
    };
    return out;
}
Tensor silu(const Tensor& a) {
    auto out = make_op(a.shape(), {a.n}, "silu");
    for (int i = 0; i < a.numel(); ++i) {
        float s = 1.f / (1.f + std::exp(-a.data()[i]));
        out.data()[i] = a.data()[i] * s;
    }
    Node *A = a.n.get(), *O = out.n.get();
    out.n->backward_fn = [A, O]() {
        for (size_t i = 0; i < O->grad.size(); ++i) {
            float x = A->data[i];
            float s = 1.f / (1.f + std::exp(-x));
            // d/dx (x*sigmoid(x)) = sigmoid + x*sigmoid*(1-sigmoid)
            A->grad[i] += (s + x * s * (1 - s)) * O->grad[i];
        }
    };
    return out;
}

}  // namespace ag
