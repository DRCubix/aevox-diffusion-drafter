#!/usr/bin/env python3
"""Generate the results figure for the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
AEVOX = "#5b8def"

# --- panel 1: acceptance (measured) ---
names = ["Unaligned\n3B", "120B MTP\n(baseline)", "Aligned 3B\n(this work)"]
acc = [2.26, 2.75, 2.79]
colors = ["#b8b8b8", "#9aa0a6", AEVOX]
b = ax1.bar(names, acc, color=colors, edgecolor="black", linewidth=0.6)
ax1.axhline(2.75, ls="--", c="#9aa0a6", lw=1)
for r, v in zip(b, acc):
    ax1.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontweight="bold")
ax1.set_ylim(2.0, 2.95)
ax1.set_ylabel("accepted tokens / target forward")
ax1.set_title("Acceptance (measured, nfe=1)")

# --- panel 2: projected speedup by drafter precision ---
prec = ["bf16\n(r=0.55)", "fp8\n(r=0.23)", "NVFP4\n(r=0.11)"]
sp = [1.8, 2.3, 2.5]
b2 = ax2.bar(prec, sp, color=["#cdd6e4", "#8fb0ec", AEVOX], edgecolor="black", linewidth=0.6)
ax2.axhline(1.41, ls="--", c="#9aa0a6", lw=1)
ax2.text(2.4, 1.45, "MTP measured 1.41x", ha="right", fontsize=8, c="#666")
for r, v in zip(b2, sp):
    ax2.text(r.get_x()+r.get_width()/2, v+0.03, f"~{v:.1f}x", ha="center", fontweight="bold")
ax2.set_ylim(0, 3.0)
ax2.set_ylabel("projected decode speedup vs AR")
ax2.set_title("Projected speedup (accept=2.79)")

fig.suptitle("AeVox Diffusion Drafter — Nemotron-3-Super-120B  ·  Daniel Rodd / AeVox.Ai",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/work/aevox-diffusion-drafter/assets/results.png", dpi=140)
print("wrote results.png")
