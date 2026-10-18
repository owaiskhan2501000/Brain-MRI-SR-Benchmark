"""Re-computes the numbers reported in main.tex from the authors' released data
(per_image_metrics.csv, manifest.json) and from the per-split accuracies of Table 6.
Run:  python scripts/verify_numbers.py <repo root containing per_image_metrics.csv and manifest.json>
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else "."
df = pd.read_csv(f"{D}/per_image_metrics.csv")
man = json.load(open(f"{D}/manifest.json"))
M = ["Bicubic", "EDSR", "Waifu2x", "RealESRGAN", "SwinIR"]
P = {m: df[df.model == m].set_index("id").sort_index() for m in M}
ids = P["Bicubic"].index
print("n images per model:", {m: len(P[m]) for m in M})

print("\n== Table 1 (mean ± SD, ddof=1) ==")
for m in M:
    d = P[m]
    print(f"{m:11s}", " | ".join(f"{c}: {d[c].mean():.3f}±{d[c].std(ddof=1):.3f}" for c in
          ["PSNR_mask", "SSIM_mask", "LPIPS", "PSNR_full", "SSIM_full"]))
print("SwinIR SSIM_mask SD (unrounded):", P["SwinIR"]["SSIM_mask"].std(ddof=1))

print("\n== Table 2 (paired diffs; Wilcoxon; bootstrap CI) ==")
rng = np.random.default_rng(0)
for m in M[1:]:
    for c in ["PSNR_mask", "SSIM_mask", "LPIPS"]:
        d = (P[m][c] - P["Bicubic"][c]).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        p = stats.wilcoxon(d).pvalue
        print(f"{m:11s}{c:10s} diff={d.mean():+.4f} CI=[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] p={p:.3g}")

print("\n== SwinIR vs EDSR (paired Wilcoxon) ==")
for c in ["PSNR_mask", "SSIM_mask", "LPIPS"]:
    d = (P["SwinIR"][c] - P["EDSR"][c]).values
    print(f"{c:10s} mean diff={d.mean():+.4f}  p={stats.wilcoxon(d).pvalue:.3g}")

print("\n== Win fractions ==")
sw, ed = P["SwinIR"]["PSNR_mask"], P["EDSR"]["PSNR_mask"]
print("SwinIR>EDSR PSNR_mask: %.1f%%" % (100 * (sw > ed).mean()))
for m in ["EDSR", "Waifu2x", "SwinIR"]:
    print(f"{m} > bicubic on all slices:", bool((P[m]["PSNR_mask"] > P["Bicubic"]["PSNR_mask"]).all()))
print("RealESRGAN < bicubic PSNR_mask: %.1f%%" % (100 * (P["RealESRGAN"]["PSNR_mask"] < P["Bicubic"]["PSNR_mask"]).mean()))
print("EDSR LPIPS better than RealESRGAN: %.1f%%" % (100 * (P["EDSR"]["LPIPS"] < P["RealESRGAN"]["LPIPS"]).mean()))
best = pd.DataFrame({m: P[m]["PSNR_mask"] for m in M}).idxmax(axis=1).value_counts()
print("best-method counts (PSNR_mask):", best.to_dict())

print("\n== Per-class means ==")
for cl in ["glioma", "meningioma", "notumor", "pituitary"]:
    for c in ["PSNR_mask", "SSIM_mask", "LPIPS"]:
        print(f"{cl:10s}{c:10s}", " ".join(f"{m}={P[m][P[m]['class']==cl][c].mean():.3f}" for m in M))

print("\n== Background / size statistics ==")
b = P["Bicubic"]
print("bg fraction mean=%.3f min=%.2f max=%.2f" % (b.bg_fraction.mean(), b.bg_fraction.min(), b.bg_fraction.max()))
print(b.groupby("class").bg_fraction.mean().round(3).to_dict())
print("full-minus-mask PSNR gap per method:", {m: round((P[m].PSNR_full - P[m].PSNR_mask).mean(), 2) for m in M})
print("manifest keys:", list(man[0].keys()))
mm = pd.DataFrame(man).set_index("id").loc[ids]
for k in ("h", "w", "height", "width", "H", "W"):
    pass
cols = [c for c in mm.columns if c.lower() in ("h", "w", "height", "width", "hr_h", "hr_w", "hr_size", "size", "shape")]
print("size-like columns:", cols)
if cols:
    print(mm[cols].head(3))
