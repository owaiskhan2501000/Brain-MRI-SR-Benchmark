"""Regenerate Fig. 2 (box plots) and Fig. 3 (histogram) from per_image_metrics.csv.
Usage: python scripts/make_figs23.py per_image_metrics.csv figures
Only labels/fonts differ from the original figures: 'Brain-mask' -> 'Head-mask' and larger text."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
order = ["Bicubic","EDSR","Waifu2x","RealESRGAN","SwinIR"]
lab = {"Bicubic":"Bicubic","EDSR":"EDSR","Waifu2x":"Waifu2x","RealESRGAN":"Real-ESRGAN","SwinIR":"SwinIR"}
col = {"Bicubic":"#8c8c8c","EDSR":"#1f77b4","Waifu2x":"#2ca02c","RealESRGAN":"#d62728","SwinIR":"#9467bd"}
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":15})

# ---- Fig 2
fig, axs = plt.subplots(1,3, figsize=(11.86,4.4))
spec = [("PSNR_mask","Head-mask PSNR (dB) \u2191","(a)"),("SSIM_mask","Head-mask SSIM \u2191","(b)"),("LPIPS","LPIPS \u2193","(c)")]
for ax,(m,yl,pl) in zip(axs,spec):
    data = [d[d.model==k][m].values for k in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, whis=1.5,
                    medianprops=dict(color="black",lw=2), whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3), boxprops=dict(lw=1.2))
    for p,k in zip(bp["boxes"],order):
        p.set_facecolor(col[k]); p.set_alpha(0.55)
    ax.set_xticks(range(1,6)); ax.set_xticklabels([lab[k] for k in order], rotation=25, ha="right", style="italic", fontsize=14)
    ax.set_ylabel(yl, fontsize=15); ax.grid(axis="y", ls=":", alpha=.6)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(-0.02,1.04,pl,transform=ax.transAxes,fontsize=17,fontweight="bold",va="bottom",ha="right" if False else "left")
fig.tight_layout()
fig.savefig(f"{out}/fig2_boxplots.png", dpi=300); plt.close(fig)

# ---- Fig 3
fig, ax = plt.subplots(figsize=(8.36,4.6))
lo, hi = d.PSNR_mask.min(), d.PSNR_mask.max()
bins = np.linspace(lo, hi, 55)
for k in order:
    ax.hist(d[d.model==k].PSNR_mask, bins=bins, alpha=.45, color=col[k], edgecolor=col[k], lw=0.8, label=lab[k])
ax.set_xlabel("Head-mask PSNR (dB)", fontsize=16); ax.set_ylabel("Number of slices", fontsize=16)
ax.grid(axis="y", ls=":", alpha=.6); ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=14)
lg = ax.legend(title="Method", fontsize=14, title_fontsize=14, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{out}/fig3_histogram.png", dpi=300); plt.close(fig)
