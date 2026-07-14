# scratch — 依存ゼロのフルスクラッチ YOLO

*[English](README.en.md) | 日本語*

**外部ライブラリを一切使わず（標準ライブラリのみ）**、自作の自動微分エンジンだけで
YOLO風の物体検出器を学習させる実装。LibTorch も BLAS も Eigen も使わない。
「学習アルゴリズムを自分の手で完全に理解する」ための教材。

## 検出結果

![results](../assets/results.png)

自作autogradだけで学習した mini-YOLOv5 の検出結果（緑=検出, 赤=正解GT）。
coco128 の person/car を 128px・2500ステップ学習（loss 0.88）。
左: 人 (conf 0.95) / 中: 複数人 / 右: 車 (conf 0.86)。
入力が128pxなので粗いが、これがモデルが実際に見ている解像度そのもの。

## 中身

| ファイル | 内容 |
|----------|------|
| `autograd.h/.cpp` | **自作autogradエンジン**。テンソル＋計算グラフ＋逆伝播。<br>ops: add/sub/mul/matmul/sum/mean、conv2d/maxpool2d/upsample/**batchnorm2d**、cat/slice、relu/sigmoid/silu |
| `net.h/.cpp` | 小型YOLO（**Conv+BN+SiLU** ×4 + **FPN 2スケール検出ヘッド**）、合成/ディスク両対応データ、YOLO風損失（サイズ別スケール割当）、NMS、学習ループ |
| `stb_image*.h` / `stb_impl.cpp` | 画像I/O専用（JPG/PNG/BMP）。単一ヘッダ・パブリックドメイン。**学習中核は不使用**（PPMは純C++のまま） |
| `main.cpp` | `test`=数値勾配チェック / `train`=学習デモ |

## ビルド & 実行

```powershell
cd scratch
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

build/Release/scratch_yolo.exe             # 勾配チェック（自作backwardの正しさを数値微分で検証）
build/Release/scratch_yolo.exe train       # 学習→検出（メモリ内の合成データ）
build/Release/scratch_yolo.exe train DIR   # 学習→検出（ディスクのPPMデータセット）
```

### データの読み込み（JPG/PNG/PPM）

画像I/Oは **stb_image / stb_image_write**（単一ヘッダ・パブリックドメイン）を使い、
**JPG / PNG / BMP を直接読み込み・PNG可視化を直接書き出し**できる。PPM は純C++ローダのままなので、
「完全依存ゼロ」で回したいときは PPM を使えばよい（**学習アルゴリズム本体は常に標準ライブラリのみ**）。

ラベルは LibTorch版と同じ YOLO形式（`class cx cy w h`）。よって `images/` + `labels/` を並べた
**同じデータフォルダを両版で共用できる**（重みは別アーキなので共用不可、データのみ共用）。

```powershell
# 例: coco128 から person/car を抽出（JPGコピー＋ラベル）
python tools/prep_coco_pb.py --src data/coco128 --out data/coco_pb
# scratch版で学習（JPGを直読み、nc=2, 128px）
scratch/build/Release/scratch_yolo.exe train data/coco_pb 2 128
```

## 仕組み（ここが学習ポイント）

### 1. 自動微分（autograd）
各テンソルは「値 `data`」「勾配 `grad`」「自分を作った演算の逆伝播関数 `backward_fn`」
「親ノード」を持つ（`Node`）。順伝播で計算グラフが自然に構築され、`loss.backward()` は
グラフを**逆トポロジカル順**にたどって各 `backward_fn` を呼び、連鎖律で勾配を親へ加算する。
これは PyTorch の autograd と同じ原理をミニマルに再現したもの。

### 2. 正しさの保証（gradient check）
自作した各演算の解析的勾配（backward）を、**中心差分による数値微分**と比較。
全演算で相対誤差 ~1e-3 以下＝実装が正しいことを数値的に証明してから積み上げている。

### mini-YOLOv5（①アンカー ②CIoU ③C3/SPPF、すべて依存ゼロ）
本格的にYOLOv5系へ寄せた構成も実装済み：
- **①アンカー**：スケールあたり3アンカー、比マッチング(<4)、v5式デコード
  （xy = 2σ-0.5+grid、wh = (2σ)²·anchor）。ヘッドは field-major 配置（ch = field·na + a）。
- **②CIoU損失**：`maximum/minimum/divide/sqrt/atan/clamp_min` を自作（各々数値微分で検証）し、
  それらを組み合わせて微分可能なCIoUを構築。
- **③C3 / SPPF backbone**：stride-2 conv ダウンサンプリング＋C3（CSP）＋SPPF＋FPN＝v5相当の構造。

これで「部品も検出方式もYOLOv5系」になる。ただし26層級のネットなので**十分な学習には多くのステップが必要**
（500ステップ・128pxでは大きい物体はクリーン検出、混雑・小物体は過検出＝学習予算/objectness較正の問題で
アーキの問題ではない）。学習の要点は本文の他節と同じ。

### 3. BatchNorm と複数スケール（FPN）
- **BatchNorm2d** も自作（順伝播の統計量計算＋非自明な逆伝播）。学習時はバッチ統計、
  推論時は running 統計に切り替え。勾配は数値微分で検証済み。
- **2スケール検出**：stride 8（8×8グリッド、小物体）と stride 16（4×4、大物体）。
  stride16特徴を upsample して stride8 に融合する **FPN トップダウン** 構造。
  各GTは**サイズで担当スケールを振り分け**（YOLOのアンカーマッチングの考え方）。

### 4. YOLO風の損失
各スケールのグリッドの各セルが `[tx,ty,tw,th, obj, cls...]` を予測。
- box: 物体セルのみ、中心オフセット＋w,h を回帰
- obj: 物体らしさ。**正例/負例を別々に正規化**して前景/背景の不均衡を回避（重要）
- cls: 物体セルのみクラス回帰

### 5. 学習と後処理
モメンタム付きSGD＋**cosine減衰**を手書き。推論時は重複枠を **NMS** で除去。
> 注意（学びどころ）：BN入りで層が深くなると学習率を下げないと発散気味になる。
> lr0=0.03 + cosine減衰 + 500ステップで安定して収束（loss 7.7→0.05）。

### 6. 高速化（依存ゼロのまま ~10倍）
畳み込みを **im2col + GEMM** に再定式化：受容野をパッチ行列に展開し、1回の行列積
`W(O,K) @ col(K,P)` にする（順・逆伝播とも）。GEMMはループ順を `i-p-j` にして内側を
連続アクセスにし、`/arch:AVX2 /fp:fast`（純コンパイラフラグ）で自動ベクトル化。
さらに **conv2d をバッチ方向でマルチスレッド化**（`std::thread`のみ）。順伝播は画像ごとに
独立、逆伝播は dW/dBias をスレッドローカルに集計→ロックして合算（dInputは画像ごとに独立）。

累積の効果（500ステップ）:

| 版 | 時間 | /step | 対naive |
|---|---|---|---|
| naive畳み込み | ~1500s | ~3s | 1× |
| im2col+GEMM+AVX2 | 157.6s | 0.315s | ~10× |
| ＋マルチスレッド | **86.9s** | **0.173s** | **~17×** |

正しさは数値微分で再検証済み。外部ライブラリ（BLAS等）は一切使っていない。
> スレッド化の効果は ~1.8× 止まり：バッチ12で並列度が低く、conv以外（BN/maxpool/損失）は
> 逐次なため（アムダールの法則）。さらに伸ばすならスレッドプール化やバッチ拡大。

## LibTorch版との違い

| | LibTorch版（親ディレクトリ） | scratch版（ここ） |
|---|---|---|
| 依存 | LibTorch必須 | **なし（標準ライブラリのみ）** |
| 規模 | YOLOv5n フル（1.9M params） | 小型（教材サイズ） |
| 速度 | 中（最適化済みカーネル） | 遅い（素朴なループ実装） |
| 目的 | 実用＋学習 | **理解・学習に全振り** |

素朴な畳み込みループなので実データ学習には向かないが、「なぜ学習が進むのか」を
1行ずつ追える。これがこのディレクトリの価値。
