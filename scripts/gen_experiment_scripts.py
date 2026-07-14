#!/usr/bin/env python3
"""Generate run scripts for the experimental settings.

Modes:
  same     for each (dataset, LLM): train and test on that dataset.
  lodo     leave-one-dataset-out: for each held-out dataset D, train on all the
           others (same LLM) and evaluate zero-shot on D.
  ablate   fixed dataset, sweep fusion / views / backbone -- the tables that
           justify the architecture.

Usage:
    python scripts/gen_experiment_scripts.py --mode lodo --llm llama3_8b
    bash scripts/generated/lodo/run_all.sh
"""

from __future__ import annotations

import argparse
import stat
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "scripts" / "generated"

DATASETS = ["triviaqa", "hotpotqa", "hotpotqa_with_context", "imdb", "movies"]
LLMS = ["llama3_8b", "mistral_7b", "qwen2.5_7b"]

HEADER = "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../../..\"\n\n"


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def gen_same(llms: list[str], datasets: list[str]) -> list[Path]:
    out = []
    for llm in llms:
        for ds in datasets:
            p = GEN / "same" / f"{llm}_{ds}.sh"
            write(p, (
                f"# Train and test on {ds} ({llm}).\n"
                f"python main.py --config configs/{ds}/{llm}.yaml train \\\n"
                f"  --run-name same_{llm}_{ds}\n"
            ))
            out.append(p)
    return out


def gen_lodo(llms: list[str], datasets: list[str]) -> list[Path]:
    """Train on N-1 datasets, test zero-shot on the held-out one."""
    out = []
    for llm in llms:
        for held_out in datasets:
            others = [d for d in datasets if d != held_out]
            p = GEN / "lodo" / f"{llm}_holdout_{held_out}.sh"
            write(p, (
                f"# Leave-one-dataset-out: train on {'+'.join(others)},\n"
                f"# then evaluate ZERO-SHOT on the unseen {held_out}.\n"
                f"# The config is only used for its LLM/model settings; the\n"
                f"# --train-datasets flag decides what is actually loaded.\n"
                f"python main.py --config configs/{held_out}/{llm}.yaml train \\\n"
                f"  --train-datasets {','.join(others)} \\\n"
                f"  --test-dataset {held_out} \\\n"
                f"  --run-name lodo_{llm}_holdout_{held_out}\n"
            ))
            out.append(p)
    return out


def gen_ablate(llms: list[str], datasets: list[str]) -> list[Path]:
    """The ablations that decide whether the architecture's claims hold."""
    out = []
    for llm in llms:
        for ds in datasets:
            base = f"python main.py --config configs/{ds}/{llm}.yaml train"

            lines = [
                "# --- Which VIEW carries the signal? ---",
                "# If V alone matches QKV, the signal is in what gets RETRIEVED,",
                "# not in what gets attended to. That would be a real finding.",
            ]
            for views in ("[Q]", "[K]", "[V]", "[Q,K]", "[Q,K,V]"):
                tag = views.strip("[]").replace(",", "")
                lines.append(
                    f"{base} \\\n  --set extract.views='{views}' "
                    f"\\\n  --run-name abl_views_{tag}_{llm}_{ds}\n"
                )

            lines.append("# --- Does the learned fusion beat plain concatenation? ---")
            lines.append("# If gated does not beat concat_mlp, say so.")
            for fusion in ("gated", "concat_mlp", "bilinear", "cross_attn"):
                lines.append(
                    f"{base} \\\n  --set model.fusion={fusion} "
                    f"\\\n  --run-name abl_fusion_{fusion}_{llm}_{ds}\n"
                )

            lines.append("# --- Do the delta channels earn their keep? ---")
            lines.append("# (raw-only is the control; boundary handling is the knob)")
            for bm in ("zero", "replicate", "wrap"):
                lines.append(
                    f"{base} \\\n  --set extract.boundary_mode={bm} "
                    f"\\\n  --run-name abl_boundary_{bm}_{llm}_{ds}\n"
                )

            lines.append("# --- Backbone: scratch vs pretrained, tied vs untied ---")
            lines.append(
                f"{base} \\\n  --set model.backbone=resnet18 "
                f"\\\n  --run-name abl_backbone_resnet18_{llm}_{ds}\n"
            )
            lines.append(
                f"{base} \\\n  --set model.share_backbone=true "
                f"\\\n  --run-name abl_shared_backbone_{llm}_{ds}\n"
            )

            p = GEN / "ablate" / f"{llm}_{ds}.sh"
            write(p, "\n".join(lines))
            out.append(p)
    return out


MODES = {"same": gen_same, "lodo": gen_lodo, "ablate": gen_ablate}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--llm", action="append", choices=LLMS,
                    help="restrict to these LLMs (default: all)")
    ap.add_argument("--dataset", action="append", choices=DATASETS,
                    help="restrict to these datasets (default: all)")
    args = ap.parse_args()

    llms = args.llm or LLMS
    datasets = args.dataset or DATASETS
    modes = list(MODES) if args.mode == "all" else [args.mode]

    for mode in modes:
        scripts = MODES[mode](llms, datasets)

        run_all = GEN / mode / "run_all.sh"
        body = f"# Runs every {mode} experiment, sequentially.\n\n" + "\n".join(
            f'echo "=== {s.stem} ==="\nbash "$(dirname "$0")/{s.name}"\n'
            for s in scripts
        )
        run_all.parent.mkdir(parents=True, exist_ok=True)
        run_all.write_text(HEADER + body)
        run_all.chmod(run_all.stat().st_mode | stat.S_IEXEC)

        print(f"{mode}: {len(scripts)} scripts -> {GEN / mode}/  (run_all.sh)")


if __name__ == "__main__":
    main()
