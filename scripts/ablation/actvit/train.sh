#!/bin/bash
# ============================================================================
# ACT-ViT ablations -- TRAIN commands. Anchor: triviaqa x llama3_8b.
#
# Run scripts/ablation/actvit/extract.sh (or the main extract) first.
# Copy lines one at a time into main.slurm, or run the whole file.
# ============================================================================
set -x

# ── views: which view carries the signal? ────────────────────────────────────
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.views='[Q]' --run-name abl_views_Q_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.views='[K]' --run-name abl_views_K_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.views='[V]' --run-name abl_views_V_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.views='[Q,K]' --run-name abl_views_QK_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.views='[Q,K,V]' --run-name abl_views_QKV_llama3_8b_triviaqa

# ── fusion: does learned fusion beat concatenation? (concat_mlp is control) ──
python main.py --config configs/triviaqa/llama3_8b.yaml train --set model.fusion=gated --run-name abl_fusion_gated_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set model.fusion=concat_mlp --run-name abl_fusion_concat_mlp_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set model.fusion=bilinear --run-name abl_fusion_bilinear_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set model.fusion=cross_attn --run-name abl_fusion_cross_attn_llama3_8b_triviaqa

# ── boundary_mode: do the delta channels earn their keep? ────────────────────
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.boundary_mode=zero --run-name abl_boundary_zero_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.boundary_mode=replicate --run-name abl_boundary_replicate_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set extract.boundary_mode=wrap --run-name abl_boundary_wrap_llama3_8b_triviaqa

# ── backbone: pretrained resnet18 (upscaled to 224x224), and tied CNNs ───────
python main.py --config configs/triviaqa/llama3_8b.yaml train --set model.backbone=resnet18 --run-name abl_backbone_resnet18_llama3_8b_triviaqa
python main.py --config configs/triviaqa/llama3_8b.yaml train --set model.share_backbone=true --run-name abl_shared_backbone_llama3_8b_triviaqa
