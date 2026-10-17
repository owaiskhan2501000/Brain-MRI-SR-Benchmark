# Benchmarking Off-the-Shelf Deep Learning Super-Resolution Models on Public Brain Tumor MRI

### A CNN-versus-Transformer Comparison Without Fine-Tuning

[Python 3.13](https://www.google.com/search?q=https%3A%2F%2Fwww.python.org%2F)
[PyTorch 2.11](https://www.google.com/search?q=https%3A%2F%2Fpytorch.org%2F)
[Colab](https://www.google.com/search?q=https%3A%2F%2Fcolab.research.google.com%2F)
[License: MIT](https://www.google.com/search?q=https%3A%2F%2Fopensource.org%2Flicenses%2FMIT)


This repository contains the official code, evaluation protocol, data manifest, and per-image results for the paper:

> **Mohammad Owais&lt;sup&gt;1,\*&lt;/sup&gt; and Muhammad Owais&lt;sup&gt;2&lt;/sup&gt;**, *"Benchmarking Off-the-Shelf Deep Learning Super-Resolution Models on Public Brain Tumor MRI: A CNN-versus-Transformer Comparison Without Fine-Tuning,"* manuscript under review, 2026.
>
> &lt;sup&gt;1&lt;/sup&gt; Department of Computer Science, Abbottabad University of Science and Technology, Abbottabad, Pakistan
> &lt;sup&gt;2&lt;/sup&gt; Faculty of Systems Design, Tokyo Metropolitan University, Tokyo, Japan
> &lt;sup&gt;\*&lt;/sup&gt; Corresponding author: owaiskhan2501000@gmail.com

---

## 📌 Overview

Pretrained single-image super-resolution (SISR) networks built for natural photographs are now routinely applied to medical scans **without any retraining**. Whether this is a sound practice for brain magnetic resonance imaging (MRI) had not been tested under a controlled, repeatable protocol.

This repository provides a **fully reproducible benchmark of five off-the-shelf 4× SISR methods** on **2000 brain MRI slices** (500 each of glioma, meningioma, pituitary tumor, and no tumor) from the public Brain Tumor MRI Dataset. Every method receives **identical, lossless LR inputs** and is run with its **frozen public weights exactly as released**.

| Family | Method | Checkpoint | Params |
|---|---|---|---|
| Interpolation (baseline) | **Bicubic** | Pillow `Image.BICUBIC` | 0 |
| CNN | **EDSR** | `edsr-base` 4× (DIV2K) | ≈ 1.5 M |
| CNN | **Waifu2x** | `nunif` UpConv-7 photo (2× cascaded twice) | ≈ 0.55 M |
| GAN | **Real-ESRGAN** | `RealESRGAN_x4plus` | ≈ 16.7 M |
| Transformer | **SwinIR** | `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` | ≈ 11.9 M |
| GAN (control) | **ESRGAN** | `ESRGAN_SRx4_DF2KOST_official` (4 slices only) | ≈ 16.7 M |

### Research questions

- **RQ1.** How well do SISR models pretrained on natural images transfer to brain MRI without fine-tuning, under a controlled bicubic degradation?
- **RQ2.** Do CNN-, GAN-, and Transformer-based architectures differ systematically in fidelity versus perceptual quality on MRI?
- **RQ3.** Does super-resolution retain the information a downstream diagnostic task depends on?

### Key message

> **Training objective matters far more than architecture.** Pixel-loss CNNs and Transformers transfer to brain MRI without fine-tuning and recover 39–61% of the class-relevant classification accuracy lost to downsampling. The perception-oriented GAN (Real-ESRGAN) trades fidelity for plausible-looking texture, fabricates high-frequency structure along tissue boundaries, and delivers an erratic downstream benefit. **Perceptual metrics alone can hide task-relevant fidelity loss** — MRI SR studies should report anatomy-restricted distortion metrics alongside perceptual ones.

---

## 🧩 Contributions

1. **Reproducible benchmark** of five off-the-shelf 4× SISR methods on 2000 public brain tumor MRI slices across four diagnostic classes, with the exact slice list, LR/HR generation script, and evaluation code released.
2. **Head-mask-restricted evaluation protocol** (PSNR and SSIM inside a foreground head mask; full-frame LPIPS), reported per class, with paired Wilcoxon signed-rank tests, bootstrap 95% CIs, and Bonferroni correction.
3. **Residual-map analysis** (|SR − GT|) that localizes each model's errors inside tumor regions and exposes synthesized texture in GAN outputs, complemented by a **bicubic-trained GAN control (ESRGAN)** that separates the adversarial loss from degradation mismatch.
4. **Downstream tumor-classification experiment** (ResNet-18) repeated over **five random training/test splits**, tying SR fidelity to task performance.
5. **Perceptual-hash (pHash) duplicate screen** quantifying the known redundancy of the source dataset (4.2% near-duplicates) and bounding its potential influence on the results.

---

## 📂 Repository Structure

```
Brain-MRI-SR-Benchmark/
├── SR_Master_Pipeline.ipynb      # Complete end-to-end pipeline (Steps 0–9, see below)
├── manifest.json                 # Exact list of the 2000 slices (500/class), dimensions & class labels
├── per_image_metrics.csv         # Per-image head-mask PSNR/SSIM, full-frame PSNR/SSIM, LPIPS
│                                 #   for all 5 methods × 2000 slices (10000 rows)
├── table1_overall.csv            # Table 1 – overall mean ± SD (head-mask + full-frame)
├── table2_paired.csv             # Table 2 – paired Δ vs bicubic, 95% bootstrap CI, Wilcoxon p
├── table3_perclass.csv           # Table 3 – per-class means (glioma / meningioma / no tumor / pituitary)
├── table4_downstream.json        # Tables 5–6 – ResNet-18 accuracy per seed (42–46) and mean ± SD
├── figures/
│   ├── fig1_workflow.png         # Study workflow
│   ├── fig2_boxplots.png         # Per-slice head-mask PSNR / SSIM / LPIPS distributions
│   ├── fig3_histogram.png        # Head-mask PSNR histogram
│   ├── fig4_glioma_detail.png    # Full slice, zoomed inset, and residual heat maps (glioma)
│   └── fig5_residuals_allclasses.png  # Residual maps for all four classes
├── LICENSE                       # MIT License for benchmark code and scripts
└── README.md
```

> **Note:** Super-resolved output images and model weights are *not* stored in this repository because of size. They are regenerated deterministically by the notebook from the public dataset and the public checkpoints listed above.

---

## 🗂️ Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.google.com/search?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fmasoudnickparvar%2Fbrain-tumor-mri-dataset) (Kaggle, [doi:10.34740/KAGGLE/DSV/2645886](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.34740%2FKAGGLE%2FDSV%2F2645886)) — 7023 JPEG slices aggregated from figshare (Cheng et al.), SARTAJ, and Br35H; axial/coronal/sagittal; four classes.
- **Benchmark subset:** 500 slices per class drawn at random from the *training* partition (seed = 42) → **n = 2000**. The exact file list is in `manifest.json`.
- **HR reference:** each slice converted to 8-bit grayscale and **mod-cropped** to the largest multiple of 32 px (so 4× and SwinIR's 8-px window divide exactly, with no padding). Because the source images are JPEG-compressed, the HR references carry native JPEG artifacts and represent the best available reference rather than an artifact-free ground truth. Side lengths range 128–1440 px; 331 slices are non-square; median side 512 px (tumor classes) vs 224 px (no-tumor).
- **LR input:** 4× **antialiased bicubic** downsampling (Pillow `Image.BICUBIC`), saved as **lossless PNG**. One shared LR set is read by every method.
- **Duplicate screen:** 64-bit DCT pHash (`imagehash` 4.3.2) flagged **84 / 2000 slices (4.2%)** as exact-hash near-duplicates (42 pairs). Paired fidelity metrics are unaffected because each slice is scored against its own HR reference; in the downstream classifier, an expected ~27 duplicate pairs straddle a random 1600/400 train/test split, a shared upper bound across all input conditions whose exact effect on absolute accuracy was not re-measured and is noted as a study limitation.

---

## ⚙️ Evaluation Protocol

### Fidelity metrics (head-mask restricted)
Background covers 34.3% of pixels on average (0–81%) and is reconstructed near-perfectly by every method, which inflates full-frame scores. All fidelity metrics are therefore computed **inside a head mask** derived from the HR reference:

1. intensity threshold > 10 → 2. morphological closing (15×15) → 3. largest connected component → 4. 4-px border crop (= scale factor).

Because no skull-stripping is applied, this foreground head mask retains the skull and scalp alongside brain parenchyma while excluding the empty air background.

- **PSNR** (MAX = 255) over head-mask pixels.
- **SSIM** — 7×7 uniform window (scikit-image default), data range 255, SSIM map averaged over the head mask.
- Full-frame PSNR/SSIM are also recorded (they exceed masked values by 1.9–2.0 dB for every method).

### Perceptual metric
- **LPIPS** (AlexNet backbone), grayscale replicated to 3 channels, computed on the **full frame** (masking deep feature maps would corrupt boundary statistics; the constant background leaves the method ranking untouched).

### Statistics
- Per-image metrics (n = 2000 per method); mean ± SD.
- **Paired two-sided Wilcoxon signed-rank tests** vs bicubic; **95% bootstrap CIs** (2000 resamples) for mean paired differences.
- **Bonferroni correction** for 15 tests (α ≈ 0.0033) — all p-values were < 10⁻²⁸ (reported as < 0.001).
- Downstream: paired two-sided t-tests across five splits (descriptive, low-powered).

### Qualitative residual analysis
For one representative slice per class: zoomed 96×96 inset on the tumor (or corresponding region), absolute residual |SR − GT| heat map on a shared 0–60 gray-level scale, and inset MAE.

### Downstream classification
- ImageNet-pretrained **ResNet-18**, 4-class, input resized to 224×224 via bilinear interpolation and normalized to [−1, 1] via (x − 0.5)/0.5, batch 32, AdamW, lr 1e-4, cross-entropy, **3 epochs**, trained on **1600 HR slices only**.
- Evaluated on the **400 held-out slices** in HR form and in each of the five SR forms (same test images per split).
- Repeated for **five stratified random splits (seeds 42–46)**; sample mean ± SD (ddof = 1) reported.

### Hardware / software
Google Colab, **NVIDIA Tesla T4 (16 GB)**, Python 3.13, PyTorch 2.11, CUDA 12.8.

---

## 🚀 How to Reproduce

The entire pipeline lives in a single notebook, **`SR_Master_Pipeline.ipynb`**, organized into sequential steps. Open it in Google Colab (GPU runtime) and run top-to-bottom.

| Step | Section | What it does |
|---|---|---|
| **0** | Dataset preparation | Download Kaggle dataset, stratified sampling (seed 42), grayscale + mod-crop → HR PNG, 4× antialiased bicubic → LR PNG, write `manifest.json`, run pHash duplicate screen |
| **1** | Bicubic | Pillow 4× upsampling baseline |
| **2** | Waifu2x | `nunif` UpConv-7 photo model, denoise off, 2× applied twice in cascade |
| **3** | EDSR | `edsr-base` 4× DIV2K checkpoint |
| **4** | Real-ESRGAN | `RealESRGAN_x4plus` checkpoint (RGB in → grayscale out) |
| **5** | SwinIR | `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4`, window 8 |
| **5b** | ESRGAN control | `ESRGAN_SRx4_DF2KOST_official` on the four representative qualitative slices |
| **6** | Quantitative evaluation | Head-mask PSNR/SSIM, full-frame PSNR/SSIM, LPIPS; Wilcoxon tests, bootstrap CIs, Bonferroni; writes `per_image_metrics.csv`, `table1`–`table3` |
| **7** | Qualitative analysis | Zoomed insets, residual heat maps (|SR − GT|), inset MAE; generates Figs. 4–5 and the ESRGAN control table |
| **8** | Downstream classification | ResNet-18 trained on 1600 HR / tested on 400 SR slices, five seeds (42–46); writes `table4_downstream.json` |
| **9** | Cost | Parameter counting and approximate per-image wall-clock inference timing on the T4 |

### Dependencies (installed by the notebook)

```
torch==2.11  torchvision  pillow  numpy  scipy  scikit-image  pandas
lpips  imagehash==4.3.2  opencv-python  matplotlib  timm  super-image
basicsr  realesrgan  (Real-ESRGAN / ESRGAN)
nunif   (Waifu2x UpConv-7)
SwinIR  (official repo, cloned in-notebook)
EDSR    (edsr-base weights, loaded in-notebook)
```

Public checkpoints are downloaded automatically from their official release pages. **No model is fine-tuned, and no post-processing or denoising is applied to any output.**

---

## 📊 Key Results

### Table 1 — Overall results on 2000 slices (4×), mean ± SD

| Method | Type | PSNR mask (dB) ↑ | SSIM mask ↑ | LPIPS ↓ | PSNR full (dB) | SSIM full |
|---|---|---|---|---|---|---|
| Bicubic | Interpolation | 28.35 ± 4.81 | 0.824 ± 0.124 | 0.240 ± 0.116 | 30.29 ± 5.25 | 0.870 ± 0.112 |
| EDSR | CNN | 31.36 ± 4.93 | 0.887 ± 0.099 | 0.142 ± 0.086 | 33.33 ± 5.33 | 0.915 ± 0.088 |
| Waifu2x (UpConv-7) | CNN | 30.67 ± 4.86 | 0.876 ± 0.104 | 0.145 ± 0.088 | 32.64 ± 5.26 | 0.908 ± 0.093 |
| Real-ESRGAN (x4plus) | GAN | 24.05 ± 3.42 | 0.752 ± 0.107 | 0.207 ± 0.072 | 26.01 ± 3.70 | 0.786 ± 0.094 |
| **SwinIR-M (classical)** | Transformer | **31.57 ± 4.95** | **0.889 ± 0.097** | **0.138 ± 0.085** | **33.53 ± 5.34** | **0.916 ± 0.087** |

### Table 2 — Paired difference from bicubic (n = 2000; 95% bootstrap CI; Wilcoxon p)

| Method | ΔPSNR (dB) | 95% CI | ΔSSIM | 95% CI | ΔLPIPS | 95% CI | p |
|---|---|---|---|---|---|---|---|
| EDSR | +3.01 | [2.97, 3.05] | +0.063 | [0.061, 0.064] | −0.099 | [−0.101, −0.097] | < 0.001 |
| Waifu2x | +2.33 | [2.30, 2.36] | +0.052 | [0.051, 0.053] | −0.095 | [−0.097, −0.093] | < 0.001 |
| Real-ESRGAN | **−4.30** | [−4.41, −4.19] | **−0.073** | [−0.074, −0.071] | −0.033 | [−0.037, −0.030] | < 0.001 |
| SwinIR | **+3.22** | [3.17, 3.27] | **+0.065** | [0.063, 0.066] | **−0.102** | [−0.104, −0.100] | < 0.001 |

- SwinIR, EDSR, and Waifu2x beat bicubic on **all 2000 slices**. SwinIR was best on 1589 slices (79.5%), EDSR on 409 (20.5%), Waifu2x on 2, and Real-ESRGAN on none.
- SwinIR–EDSR margin is small (+0.21 dB) but significant (p < 0.001 on all metrics, after Bonferroni).
- **Real-ESRGAN fell below bicubic on 93.1% of slices** in head-mask PSNR while *improving* LPIPS (0.207 vs 0.240) — a textbook perception–distortion trade-off.

### Table 3 — Per-class means (500 slices per class)

| Class | Metric | Bicubic | EDSR | Waifu2x | Real-ESRGAN | SwinIR |
|---|---|---|---|---|---|---|
| Glioma | PSNR | 30.46 | 33.39 | 32.77 | 25.36 | **33.53** |
| | SSIM | 0.874 | 0.924 | 0.916 | 0.799 | **0.925** |
| | LPIPS | 0.170 | 0.098 | 0.099 | 0.176 | **0.095** |
| Meningioma | PSNR | 30.48 | 33.38 | 32.66 | 24.82 | **33.62** |
| | SSIM | 0.880 | 0.926 | 0.917 | 0.788 | **0.927** |
| | LPIPS | 0.186 | 0.111 | 0.112 | 0.179 | **0.109** |
| No tumor | PSNR | 22.18 | 25.35 | 24.63 | 20.35 | **25.67** |
| | SSIM | 0.678 | 0.782 | 0.764 | 0.631 | **0.786** |
| | LPIPS | 0.378 | 0.219 | 0.226 | 0.237 | **0.215** |
| Pituitary | PSNR | 30.26 | 33.30 | 32.63 | 25.66 | **33.46** |
| | SSIM | 0.864 | 0.915 | 0.907 | 0.788 | **0.917** |
| | LPIPS | 0.227 | 0.139 | 0.144 | 0.235 | **0.134** |

The method ordering (SwinIR > EDSR > Waifu2x > Bicubic > Real-ESRGAN for PSNR/SSIM) is identical in every class. Lower absolute scores in the no-tumor class reflect its much smaller native image size (median 224 px vs 512 px; log-area vs bicubic PSNR Pearson r = 0.82), not model behavior.

### Table 4 — Bicubic-trained GAN control: inset MAE (gray levels, 96×96 inset)

| Slice (class) | Bicubic | EDSR | Waifu2x | SwinIR | ESRGAN | Real-ESRGAN |
|---|---|---|---|---|---|---|
| Glioma | 4.02 | **3.11** | 3.20 | 3.24 | 4.85 | 8.60 |
| Meningioma | 3.96 | 3.18 | 3.18 | **3.17** | 4.05 | 6.69 |
| No tumor | 6.43 | 4.84 | 5.05 | **4.74** | 7.03 | 10.47 |
| Pituitary | 4.55 | **3.52** | 3.58 | 3.54 | 5.32 | 7.16 |

ESRGAN (same RRDB generator as Real-ESRGAN, but trained under a *matched* bicubic degradation) removes ~70–97% of Real-ESRGAN's excess error over bicubic, yet still sits at or below bicubic fidelity → **degradation mismatch sets the magnitude of the GAN's failure; the adversarial objective sets its direction.**

### Table 5 — Downstream ResNet-18 tumor classification (mean ± SD over 5 splits, seeds 42–46)

| Input to classifier | Accuracy (%) | Change vs HR (pp) | Δ vs bicubic (pp) [p] | Recovery |
|---|---|---|---|---|
| Ground truth (HR) | 93.25 ± 1.81 | — | — | 100% |
| Bicubic | 83.95 ± 3.76 | −9.30 | — | 0% |
| EDSR | 88.95 ± 2.86 | −4.30 | +5.00 ± 1.77 [0.003] | 54% |
| Waifu2x | 87.60 ± 2.55 | −5.65 | +3.65 ± 2.20 [0.021] | 39% |
| Real-ESRGAN | 86.70 ± 2.86 | −6.55 | +2.75 ± 5.11 [0.30] | 30% |
| **SwinIR** | **89.60 ± 2.27** | **−3.65** | **+5.65 ± 1.96 [0.003]** | **61%** |

*(Note: The HR-versus-bicubic accuracy gap is +9.30 ± 3.78 pp, p = 0.005.)*

Per-split accuracies (%):

| Seed | HR | Bicubic | EDSR | Waifu2x | Real-ESRGAN | SwinIR |
|---|---|---|---|---|---|---|
| 42 | 94.75 | 86.75 | 90.75 | 88.00 | 81.75 | 91.50 |
| 43 | 92.75 | 82.00 | 89.00 | 87.25 | 87.75 | 89.50 |
| 44 | 91.25 | 87.75 | 90.25 | 89.00 | 89.00 | 90.50 |
| 45 | 92.00 | 78.50 | 84.00 | 83.50 | 87.00 | 85.75 |
| 46 | 95.50 | 84.75 | 90.75 | 90.25 | 88.00 | 90.75 |

- Downsampling + bicubic upsampling costs **9.3 pp** of accuracy, in every split.
- The three **distortion-oriented models beat bicubic in all five splits** (SwinIR p = 0.003, EDSR p = 0.003, Waifu2x p = 0.021), recovering 39–61% of the loss.
- **Real-ESRGAN's gain is small and inconsistent** (below bicubic in seed 42; p = 0.30).
- The downstream ordering tracks **head-mask PSNR/SSIM**, not LPIPS → fidelity to the reference, not perceptual sharpness, preserves class-discriminative information.

### Table 6 — Model size and approximate inference cost (Tesla T4, 16 GB)

| Method | Type | Parameters | Time / image (approx.) | Total (2000 images) |
|---|---|---|---|---|
| Bicubic | Interpolation | 0 | ≈ 0.5 ms (CPU) | ≈ 1 s |
| Waifu2x (UpConv-7) | CNN | ≈ 0.55 M | ≈ 60 ms | ≈ 2 min |
| EDSR (base) | CNN | ≈ 1.5 M | ≈ 60 ms | ≈ 2 min |
| Real-ESRGAN (x4plus) | GAN | ≈ 16.7 M | ≈ 150 ms | ≈ 5 min |
| SwinIR-M (classical) | Transformer | ≈ 11.9 M | ≈ 600 ms | ≈ 20 min |

SwinIR's +0.21 dB over EDSR costs ~8× the parameters and ~10× the inference time. EDSR and Waifu2x deliver most of the attainable gain at a fraction of the cost.

---

## 🔍 Main Findings at a Glance

| | Finding |
|---|---|
| **RQ1 — Transfer** | Under a controlled bicubic degradation, distortion-oriented SISR models pretrained on natural images transfer to brain MRI with no fine-tuning: +2.3 to +3.2 dB head-mask PSNR over bicubic on every one of 2000 slices, in all four classes. |
| **RQ2 — Architecture** | Architecture family mattered less than training objective. The CNN (EDSR) and Transformer (SwinIR), both trained on DIV2K with pixel losses, behaved almost interchangeably; the GAN (Real-ESRGAN) occupied a separate regime of lower fidelity and better—but still inferior—perceptual scores. |
| **RQ3 — Task utility** | SR preserved class-discriminative information in proportion to its fidelity: distortion-oriented models recovered 39–61% of the classification accuracy lost to downsampling, consistently across splits; Real-ESRGAN recovered 30% on average with an unreliable benefit. |
| **Protocol** | Full-frame PSNR over-estimates head-mask PSNR by ≈ 2 dB for every method; lossy LR inputs and non-antialiased downsampling produced spurious checkerboard artifacts in preliminary runs. One shared lossless LR set + head-mask metrics + paired statistics closes these traps. |

---

## ⚠️ Limitations

1. **Synthetic degradation.** LR inputs were produced by bicubic downsampling — the very degradation the distortion-oriented checkpoints were trained on. Real low-field / accelerated acquisitions lose resolution through k-space truncation, noise, and motion. Results describe transfer under an idealized degradation, not clinical performance, and the mismatch works against blind-SR models such as Real-ESRGAN.
2. **Data.** 8-bit, JPEG-derived 2D slices from a public aggregate without acquisition metadata; heterogeneous sizes; the no-tumor class is systematically smaller; no 3D volumetric SR.
3. **Duplicates.** 4.2% near-duplicates by exact pHash match (adjacent slices from the same volume would not be caught). Near-duplicate pairs do not bias the paired fidelity metrics; their exact effect on the five random downstream classification splits (~27 split-straddling pairs expected per random split) was bounded rather than directly re-measured.
4. **No fine-tuning by design.** Domain-adapted MRI SR models would be expected to do better. Waifu2x was run as a 2×→2× cascade. The ESRGAN control covers four slices and inset MAE only.
5. **Foreground head mask & full-frame LPIPS.** The intensity threshold mask retains skull and scalp alongside brain parenchyma (no skull-stripping was applied), and LPIPS was computed full-frame, making it less anatomy-specific than the masked distortion metrics.
6. **Downstream proxy.** ResNet-18 trained for 3 epochs on 1600 HR images only (introducing a potential train-versus-test domain shift when evaluated on SR outputs); considerable split-to-split variance; differences among the distortion-oriented models and vs Real-ESRGAN are only partly resolved with five splits.
7. **No radiologist reader study** — no claim about diagnostic reading is made.
8. **Single public collection** — limited scanner and protocol diversity.

---

## 🔭 Future Work

- Evaluate fine-tuned and MRI-specific SR models under the same protocol.
- Extend the ESRGAN control to the full 2000-slice benchmark.
- Replicate with **k-space-truncated** LR inputs (the most important next step).
- Re-run on a curated, duplicate-free dataset with acquisition metadata (e.g., BraTS, IXI).
- Extend to 3D volumetric SR.
- Strengthen the downstream experiment (larger test set, classifier trained to convergence and on SR outputs, more splits).
- Radiologist reader studies to establish the diagnostic impact of synthesized texture.

---

## 📝 Citation

If you use this code, the manifest, or the benchmark results, please cite:

```bibtex
@unpublished{owais2026srbenchmark,
  author = {Owais, Mohammad and Owais, Muhammad},
  title  = {Benchmarking Off-the-Shelf Deep Learning Super-Resolution Models on
            Public Brain Tumor {MRI}: A {CNN}-versus-{T}ransformer Comparison
            Without Fine-Tuning},
  note   = {Manuscript under review},
  year   = {2026},
  url    = {https://github.com/owaiskhan2501000/Brain-MRI-SR-Benchmark}
}
```

Plain text:

> Mohammad Owais and Muhammad Owais, "Benchmarking Off-the-Shelf Deep Learning Super-Resolution Models on Public Brain Tumor MRI: A CNN-versus-Transformer Comparison Without Fine-Tuning," manuscript under review, 2026.

Please also cite the dataset and the original model papers:

- **Dataset:** M. Nickparvar, *Brain Tumor MRI Dataset*, Kaggle, 2021. doi:10.34740/KAGGLE/DSV/2645886; J. Cheng et al., figshare brain tumor dataset, 2015.
- **EDSR:** B. Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution," CVPRW 2017.
- **Waifu2x / nunif:** nagadomi, *nunif* (UpConv-7), GitHub.
- **Real-ESRGAN:** X. Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," ICCVW 2021.
- **ESRGAN:** X. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks," ECCVW 2018.
- **SwinIR:** J. Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCVW 2021.
- **LPIPS:** R. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018.
- **pHash:** C. Zauner, *Implementation and Benchmarking of Perceptual Image Hash Functions*, Dissertation, 2010; `imagehash` 4.3.2.

---

## 📜 Declarations

- **Funding:** This research received no external funding.
- **Ethics:** Not applicable — only pre-existing, fully de-identified, publicly available imaging data were used; no new patient data were collected.
- **Competing interests:** The authors declare no competing interests.
- **Author contributions:** *Mohammad Owais (MoO)* — Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Visualization, Writing – original draft. *Muhammad Owais (MuO)* — Validation, Writing – review & editing.
- **Generative AI use:** A large language model (Claude, Anthropic) was used for language editing and for reviewing the experimental pipeline for implementation errors. All text, code, analyses, and conclusions were reviewed, verified, and are the sole responsibility of the authors; the tool was not used to generate, analyze, or interpret data.

## 🙏 Acknowledgements

The authors thank the creators of the Brain Tumor MRI Dataset and the authors of the open-source SR implementations (EDSR, nunif/Waifu2x, Real-ESRGAN, ESRGAN, SwinIR, LPIPS, imagehash) for making their data and code publicly available.

## 📄 License

Code and evaluation scripts in this repository are released under the **MIT License** (see `LICENSE`). The Brain Tumor MRI Dataset and the pretrained model weights remain subject to their own respective original licenses — please consult the original sources before redistribution.

---

&lt;p align=&quot;center&quot;&gt;&lt;sub&gt;Maintained by Mohammad Owais · Abbottabad University of Science and Technology · &lt;a href=&quot;mailto:owaiskhan2501000@gmail.com&quot;&gt;owaiskhan2501000@gmail.com&lt;/a&gt;&lt;/sub&gt;&lt;/p&gt;
