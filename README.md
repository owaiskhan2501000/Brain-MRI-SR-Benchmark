# Benchmarking Off-the-Shelf Deep Learning Super-Resolution Models on Public Brain Tumour MRI

This repository contains the official code, evaluation scripts, and data splits for the paper:
**"Benchmarking Off-the-Shelf Deep Learning Super-Resolution Models on Public Brain Tumour MRI: A CNN-versus-Transformer Comparison Without Fine-Tuning"**

## 📌 Overview
Deep learning single-image super-resolution (SISR) models trained on natural images are often applied to medical images without retraining. This repository provides a reproducible benchmark of five off-the-shelf 4x SISR methods on 2,000 brain MRI slices:
- **Baseline:** Bicubic Interpolation
- **CNN-based:** EDSR, Waifu2x
- **GAN-based:** Real-ESRGAN
- **Transformer-based:** SwinIR

Our findings demonstrate that distortion-oriented models (CNNs and Transformers) transfer to brain MRI reliably without fine-tuning, recovering most of the diagnostic information. Conversely, perceptually-oriented GAN models (like Real-ESRGAN) hallucinate artificial textures that reduce clinical fidelity and degrade downstream diagnostic accuracy.

## 📂 Repository Structure
- `manifest.json`: Contains the exact list of 2,000 images (500 per class: glioma, meningioma, pituitary, no_tumor) used for benchmarking.
- `per_image_metrics.csv`: Detailed evaluation metrics (PSNR, SSIM, LPIPS) for all 10,000 comparisons.
- `table1_overall.csv` to `table4_downstream.json`: Aggregated results and downstream classification accuracies.
- `Notebooks/`: Contains the Google Colab notebooks to reproduce the entire pipeline from scratch.

## 🚀 How to Reproduce
The entire pipeline is divided into sequential steps provided in the Jupyter/Colab notebooks:

- **Step 0:** Dataset preparation, 4x antialiased bicubic downsampling, and `manifest.json` creation.
- **Steps 1-5:** Model Inference (Bicubic, Waifu2x, EDSR, Real-ESRGAN, SwinIR) using frozen publicly available weights.
- **Step 6:** Quantitative Evaluation. Calculates brain-mask-restricted PSNR/SSIM and full-frame LPIPS.
- **Step 7:** Qualitative Analysis. Generates zoomed insets and absolute residual error heatmaps (|SR - GT|).
- **Step 8:** Downstream Classification. Trains a ResNet-18 classifier on 1,600 HR images and tests it on 400 super-resolved images.
- **Step 9:** Parameter counting and hardware runtime evaluation.

## 📊 Key Results
- **Fidelity:** SwinIR achieved the highest masked PSNR (31.57 dB), closely followed by EDSR (31.36 dB). Real-ESRGAN performed the worst (24.05 dB), scoring below the bicubic baseline.
- **Clinical Utility:** ResNet-18 downstream classification accuracy was 94.75% on Ground Truth, 94.00% on SwinIR, but dropped to 90.25% on Real-ESRGAN.

## 📝 Citation
If you use this code or benchmark, please cite our paper:
```text
[Add your paper citation here once published]
