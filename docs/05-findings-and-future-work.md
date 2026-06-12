# 05 — Findings & Future Work

## What we proved
1. **A diffusion LM, LoRA-aligned to a *different* target, beats that target's own MTP self-speculation** at the acceptance level (2.79 vs 2.75; 2.26 unaligned). The shared byte-identical tokenizer makes this cross-model setup valid and lossless.
2. **The acceptance ceiling for a token-only, single-step, standalone drafter is ~2.8** — and we located it with controlled experiments rather than guesses.

## Why the cheap levers don't break ~2.8 (the key insight)
We expected "more data / better objective" to keep lifting acceptance. It didn't — and the reason is mechanistic:

- **Data-saturated:** 60 examples → 2.72, 10K → 2.76. 160× more data bought +0.04. A standalone 3B learns the marginal "what follows this context" quickly, then saturates.
- **Logit distillation ≈ hard tokens here:** the 120B's per-token distribution is **extremely peaked** (top-1 prob ≈ 1.0; top-2 ≈ 0.002 on confident reasoning tokens). Soft-target KL only helps over hard labels when the teacher is *uncertain* — so KL came in at 2.17 ≤ CE 2.30.
- **Multi-step is anti-economical:** more denoising steps lift acceptance a little (2.30→2.57) but multiply draft cost (nfe 1→13), collapsing the net speedup.

**Conclusion:** the bottleneck is *information*, not optimization. A token-only drafter sees only tokens; to predict a *different* model's continuation better than ~2.8, it needs information the peaked token stream doesn't carry.

## The path to 3–4× (evidence-backed)
Two methods add exactly that missing information:

1. **Hidden-state conditioning (DFlash).** Train a small block-diffusion *head* that consumes the 120B's mid-layer **hidden states** (EAGLE-3-style), reusing the target's frozen embedding/lm-head. Published DFlash results: acceptance 6–8, up to 4.9× / 2.4× over EAGLE-3 — on transformer targets. For our hybrid-Mamba 120B this is novel (no published DFlash-on-Mamba), and needs the unreleased training recipe reimplemented, but the hidden states are **recoverable from our existing dataset via teacher-forced forwards** (no re-generation). Storage for full hidden-state caching is large (~tens of TB at 100K scale) — plan for recompute-on-the-fly or shared storage.
2. **Tree / multi-candidate drafting (DDTree, +37–53% acceptance, no retrain).** Build a draft *tree* from the diffusion drafter's per-position top-k marginals and verify it in one forward. **Caveat for our target:** tree verification through Mamba2 layers requires the **STree TreeScan** kernel (`A_tree = L·A_log`) + activation-replay rollback — standard KV tree-attention covers the attention/MoE layers but breaks/OOMs on the SSM layers. STree's overhead *shrinks* with model size, so at 120B it should be a small fraction of the verify cost.

## Recommended next steps (in order)
1. **Build the vLLM proposer** to convert the validated 2.79 acceptance into a *measured* ~2.5× served tok/s (see [04-serving.md](04-serving.md)).
2. **NVFP4-QAT the aligned drafter** to hit the `r≈0.11` cost projection.
3. **DFlash head** for the 120B (hidden-state conditioning) — the clearest lever to 3–4×; data is reusable, training needs a cluster.
4. **DDTree + STree TreeScan** — orthogonal +40% on top, the main engineering risk being the custom Mamba tree-scan kernel.

## Negative results worth publishing
- Logit-KL distillation does **not** beat hard-token CE when the teacher distribution is peaked (confident model distilling its own outputs).
- Data/epoch scaling does **not** lift a token-only standalone drafter past its mechanism limit.

These are useful for anyone attempting cross-model diffusion drafting — they save months of chasing the wrong lever.
