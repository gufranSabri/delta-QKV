#!/bin/bash
# ============================================================================
# ACT-ViT ablations -- TEST commands. Anchor: triviaqa x llama3_8b.
#
# Scores each ablation run on triviaqa's held-out corpus, appending to
# docs/results.csv. Run the matching train.sh line first.
# ============================================================================
set -x

# ── views ────────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_views_Q_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_views_K_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_views_V_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_views_QK_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_views_QKV_llama3_8b_triviaqa/best.pt --dataset triviaqa

# ── fusion ───────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_fusion_gated_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_fusion_concat_mlp_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_fusion_bilinear_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_fusion_cross_attn_llama3_8b_triviaqa/best.pt --dataset triviaqa

# ── boundary_mode ────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_boundary_zero_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_boundary_replicate_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_boundary_wrap_llama3_8b_triviaqa/best.pt --dataset triviaqa

# ── backbone ─────────────────────────────────────────────────────────────────
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_backbone_resnet18_llama3_8b_triviaqa/best.pt --dataset triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml test --checkpoint runs/abl_shared_backbone_llama3_8b_triviaqa/best.pt --dataset triviaqa
