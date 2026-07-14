#pragma once
// A tiny reverse-mode autograd engine in pure C++ (no external deps).
// Tensors are dense row-major float arrays that remember how they were
// computed, so calling backward() on a scalar populates .grad() on every
// tensor that fed into it. This is the "understand training by building it"
// core of the from-scratch YOLO.
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <cstddef>

namespace ag {

struct Node {
    std::vector<float> data;     // forward values (row-major)
    std::vector<float> grad;     // dLoss/dthis, same size as data
    std::vector<int> shape;
    std::vector<std::shared_ptr<Node>> parents;  // keep the graph alive
    std::function<void()> backward_fn;           // accumulate into parents' grad
    bool requires_grad = false;
    std::string op;

    int numel() const {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }
};
using NodePtr = std::shared_ptr<Node>;

class Tensor {
public:
    NodePtr n;
    Tensor() = default;
    explicit Tensor(NodePtr node) : n(std::move(node)) {}

    static Tensor zeros(std::vector<int> shape, bool requires_grad = false);
    static Tensor from(std::vector<float> data, std::vector<int> shape,
                       bool requires_grad = false);
    static Tensor randn(std::vector<int> shape, float std = 1.0f,
                        bool requires_grad = false);

    std::vector<int>& shape() const { return n->shape; }
    std::vector<float>& data() const { return n->data; }
    std::vector<float>& grad() const { return n->grad; }
    int numel() const { return n->numel(); }
    float item() const { return n->data[0]; }

    void backward() const;     // seed grad=1 on a scalar, run reverse-topo
    void zero_grad() const;    // zero this node's grad (used on parameters)
};

// Build a fresh output node wired to `parents`. Caller fills data + backward_fn.
Tensor make_op(std::vector<int> shape, std::vector<NodePtr> parents,
               const std::string& op);

// ---- elementwise & reductions ----
Tensor add(const Tensor& a, const Tensor& b);   // same shape
Tensor sub(const Tensor& a, const Tensor& b);   // same shape
Tensor mul(const Tensor& a, const Tensor& b);   // same shape (Hadamard)
Tensor mul_scalar(const Tensor& a, float s);
Tensor add_scalar(const Tensor& a, float s);
Tensor sum(const Tensor& a);                    // -> scalar
Tensor mean(const Tensor& a);                   // -> scalar

// ---- linear algebra ----
Tensor matmul(const Tensor& a, const Tensor& b);  // (m,k)x(k,n) -> (m,n)

// ---- conv / pooling (NCHW) ----
// x:(N,C,H,W) w:(O,C,kh,kw) b:(O) -> (N,O,OH,OW)
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride, int pad);
Tensor maxpool2d(const Tensor& x, int k, int stride, int pad);
Tensor upsample_nearest2d(const Tensor& x, int scale);
// BatchNorm over (N,C,H,W), per-channel. gamma/beta are learnable (C);
// running_mean/running_var are non-grad buffers, updated in place when training.
Tensor batchnorm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   Tensor& running_mean, Tensor& running_var,
                   bool training, float momentum = 0.1f, float eps = 1e-5f);
// add a per-channel bias (O) to (N,O,H,W)
Tensor add_bias_nchw(const Tensor& x, const Tensor& b);
// concatenate a list of (N,C_i,H,W) along channels
Tensor cat_channels(const std::vector<Tensor>& xs);
// slice channels [c0,c1) of (N,C,H,W)
Tensor slice_channels(const Tensor& x, int c0, int c1);

// ---- elementwise math (for CIoU); all same-shape ----
Tensor maximum(const Tensor& a, const Tensor& b);
Tensor minimum(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);
Tensor sqrt_(const Tensor& a);
Tensor atan_(const Tensor& a);
Tensor clamp_min(const Tensor& a, float m);

// ---- activations ----
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor silu(const Tensor& a);

// operator sugar
inline Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
inline Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }

// RNG control (deterministic, no <random> global state surprises).
void seed(uint64_t s);
float randf();  // uniform [0,1)

}  // namespace ag
