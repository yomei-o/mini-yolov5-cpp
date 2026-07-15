# scratch_py — NumPyだけで自作autograd × mini-YOLOv5（Python版）

*日本語 | [English](README.en.md)*

`scratch/`（依存ゼロC++版）の **Python移植**。ディープラーニングのフレームワーク
（PyTorch / TensorFlow）を一切使わず、**NumPyの配列演算だけ**を土台にして、
自作の自動微分エンジンで YOLO風の物体検出器を学習させる。
「学習アルゴリズムを自分の手で完全に理解する」ための教材。C++版と1行ずつ読み比べられる。

## 中身

| ファイル | 内容 |
|----------|------|
| `autograd.py` | **自作autogradエンジン**。Tensor＋計算グラフ＋逆伝播。<br>ops: add/sub/mul/matmul/sum/mean、conv2d/maxpool2d/upsample/**batchnorm2d**、cat/slice、maximum/minimum/divide/sqrt/atan/clamp_min、relu/sigmoid/silu |
| `net.py` | mini-YOLOv5（**C3/SPPF backbone + FPN 2スケール検出ヘッド**）、合成/ディスク両対応データ、**CIoU損失**、NMS、学習ループ、画像I/O（PPM読み・PNG書き） |
| `main.py` | 引数なし=数値勾配チェック / `train`=学習デモ |

**NumPy以外の依存はゼロ**（JPG/PNGを読むときだけ任意でPillow。PPMなら完全にNumPyのみ）。

## 実行

```bash
cd scratch_py
python main.py                    # 勾配チェック（自作backwardの正しさを数値微分で検証）
python main.py train              # 学習→検出（メモリ内の合成データ, nc=3, 64px）
python main.py train DIR          # 学習→検出（ディスクのデータセット, nc=3, 64px）
python main.py train DIR 2 128    # DIRを nc=2, 128px で学習
```

学習後、`viz_0.png` … に検出結果（緑=検出, 赤=正解GT）が書き出される。

### データの共用

ラベルは LibTorch版・C++ scratch版と同じ YOLO形式（`class cx cy w h`）。
`images/` + `labels/` を並べた **同じデータフォルダを全版で共用できる**
（重みは別アーキなので共用不可、データのみ）。PPMは純Pythonで読み込むので
「NumPyのみ」で回せる。JPG/PNGは Pillow があれば直読みできる。

```bash
# 例: coco128 から person/car を抽出（親ディレクトリのツール）
python tools/prep_coco_pb.py --src data/coco128 --out data/coco_pb
python scratch_py/main.py train data/coco_pb 2 128
```

## 仕組み（学習ポイント）

### 1. 自動微分（autograd）
各 `Tensor` は「値 `data`（NumPy配列）」「勾配 `grad`」「自分を作った演算の
逆伝播クロージャ `backward_fn`」「親ノード `parents`」を持つ。順伝播で計算グラフが
自然に構築され、`loss.backward()` はグラフを**逆トポロジカル順**にたどって各
`backward_fn` を呼び、連鎖律で勾配を親へ加算する。PyTorch の autograd と同じ原理。

**NumPyは「配列演算」だけ**を担当し、微分（各opのbackward）はすべて手書き。
C++版が素朴な多重ループで書いていた順・逆伝播が、ここではベクトル化された
NumPy演算になる（例：conv は `im2col` してから `np.einsum` で行列積）。

### 2. 正しさの保証（gradient check）
自作した各演算の解析的勾配（backward）を、**中心差分による数値微分**と比較。
`python main.py` で全opが相対誤差 ~1e-7 以下＝実装が正しいことを数値的に証明。

> C++版は float32（SIMD都合）だが、Python版は **float64**。NumPyの自然な精度で、
> 勾配チェックが桁落ちに埋もれず綺麗に通る（これがこのファイルの主眼）。

### 3. mini-YOLOv5（アンカー・CIoU・C3/SPPF）
本格的にYOLOv5系へ寄せた構成：
- **アンカー**：スケールあたり3アンカー、比マッチング(<4)、v5式デコード
  （xy = 2σ-0.5+grid、wh = (2σ)²·anchor）。ヘッドは field-major 配置（ch = field·na + a）。
- **CIoU損失**：`maximum/minimum/divide/sqrt/atan/clamp_min` を自作（各々数値微分で検証）し、
  それらを組み合わせて微分可能なCIoUを構築。
- **C3 / SPPF backbone**：stride-2 conv ダウンサンプリング＋C3（CSP）＋SPPF＋FPN＝v5相当の構造。

### 4. BatchNorm と複数スケール（FPN）
- **BatchNorm2d** も自作（順伝播の統計量計算＋非自明な逆伝播）。学習時はバッチ統計、
  推論時は running 統計に切り替え。勾配は数値微分で検証済み。
- **2スケール検出**：stride 8（小物体）と stride 16（大物体）。stride16特徴を
  upsample して stride8 に融合する **FPN トップダウン** 構造。各GTは
  **サイズで担当スケールを振り分け**（YOLOのアンカーマッチングの考え方）。

### 5. YOLO風の損失
各スケールのグリッドの各セルが `[tx,ty,tw,th, obj, cls...]` を予測。
- box: 物体セルのみ CIoU回帰
- obj: 物体らしさ。**正例/負例を別々に正規化**して前景/背景の不均衡を回避、
  さらに目標値を detached CIoU にして「信頼度=枠の質」に較正（誤検出を減らす）
- cls: 物体セルのみクラス回帰

### 6. 学習
モメンタム付きSGD＋**線形warmup＋cosine減衰**を手書き。推論時は重複枠を **NMS** で除去。

## C++ scratch版との違い

| | C++ scratch版（`../scratch`） | Python版（ここ） |
|---|---|---|
| 依存 | なし（標準ライブラリのみ） | **NumPyのみ**（画像はPPMなら純Python） |
| 数値精度 | float32 | float64 |
| conv実装 | 手書き im2col+GEMM（+AVX2/マルチスレッド） | im2col + `np.einsum` |
| 速度 | 最適化済みで速い | NumPy任せ（十分実用、C++最適化版よりは遅い） |
| 目的 | 依存ゼロを徹底 | **Pythonで最短・最も読みやすく** |

アルゴリズムは完全に同一。片方を読んで詰まったらもう片方を見れば、
「C++の多重ループ」と「NumPyのベクトル化」がそのまま対応していることが分かる。
