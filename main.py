#!/usr/bin/env python3
"""delta-QKV: hallucination detection from per-token Q/K/V activation images.

Subcommands:
    extract   generate responses, capture Q/K/V, build and save token images
    label     recompute labels from stored responses (no re-extraction needed)
    train     train the detector (single dataset, or leave-one-dataset-out)
    test      evaluate a saved checkpoint
    inspect   render token images to PNG so you can actually look at them
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_config
from src.utils.logger import setup_logging


def _overrides(args) -> dict:
    """Turn --set a.b=c flags into a nested override dict."""
    out: dict = {}
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)

        # Parse the value with YAML so ints/floats/bools/lists come through typed.
        import yaml

        parsed = yaml.safe_load(value)

        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return out


def main(argv=None) -> int:
    # --config and --set are declared on a shared parent parser AND inherited by
    # every subcommand, so they work on either side of the subcommand name:
    #     main.py --config c.yaml train --set model.fusion=gated
    #     main.py train --config c.yaml --set model.fusion=gated
    # argparse otherwise binds a top-level flag only before the subcommand, which
    # is a trap: the natural `train --set ...` ordering would just error out.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to a YAML config")
    common.add_argument(
        "--set", action="append", metavar="KEY=VAL", default=[],
        help="override any config key, e.g. --set model.fusion=bilinear "
             "--set extract.views='[V]'",
    )

    parser = argparse.ArgumentParser(
        prog="delta-QKV",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", parents=[common],
                          help="generate + capture Q/K/V + save images")
    p_ex.add_argument("--chunk", type=int, default=None,
                      help="1-indexed block of 1000 examples (for splitting a long run)")
    p_ex.add_argument("--overwrite", action="store_true",
                      help="re-extract examples that already have tokens.npy")

    sub.add_parser("label", parents=[common],
                   help="recompute labels from stored responses")

    p_tr = sub.add_parser("train", parents=[common], help="train the detector")
    p_tr.add_argument("--train-datasets", type=str, default=None,
                      help="comma-separated. Defaults to the config's dataset.")
    p_tr.add_argument("--test-dataset", type=str, default=None,
                      help="held-out dataset for zero-shot evaluation")
    p_tr.add_argument("--run-name", type=str, default=None)

    p_te = sub.add_parser("test", parents=[common], help="evaluate a checkpoint")
    p_te.add_argument("--checkpoint", required=True)
    p_te.add_argument("--dataset", default=None, help="defaults to the config's dataset")

    p_in = sub.add_parser("inspect", parents=[common], help="render token images to PNG")
    p_in.add_argument("--idx", type=int, default=0, help="example index")
    p_in.add_argument("--tokens", type=int, default=4, help="how many tokens to render")
    p_in.add_argument("--out", default=None)

    # Two-stage parse. When a flag is declared on BOTH the top-level parser and a
    # subparser (via `parents`), argparse runs the subparser LAST, so its default
    # silently overwrites whatever the top-level flag captured -- i.e.
    # `--config c.yaml train` would end up with config=None. Parsing the
    # pre-subcommand args separately and then filling in any gaps avoids that,
    # and lets both orderings work.
    pre, _ = common.parse_known_args(argv)
    args = parser.parse_args(argv)

    if not args.config:
        args.config = pre.config
    # Merge, don't replace: --set may legitimately appear on both sides.
    args.set = list(pre.set or []) + [s for s in (args.set or []) if s not in (pre.set or [])]

    if not args.config:
        parser.error("--config is required (before or after the subcommand)")
    setup_logging()

    cfg = load_config(args.config, overrides=_overrides(args))

    if args.cmd == "extract":
        from src.extract.run_extraction import run_extraction

        run_extraction(cfg, chunk=args.chunk, overwrite=args.overwrite)

    elif args.cmd == "label":
        from src.extract.run_extraction import relabel

        relabel(cfg)

    elif args.cmd == "train":
        from src.train import train

        train_datasets = (
            [d.strip() for d in args.train_datasets.split(",") if d.strip()]
            if args.train_datasets
            else [cfg.dataset.name]
        )
        train(cfg, train_datasets, args.test_dataset, run_name=args.run_name)

    elif args.cmd == "test":
        from src.test import test

        test(cfg, args.checkpoint, dataset_name=args.dataset)

    elif args.cmd == "inspect":
        from src.inspect_images import inspect

        inspect(cfg, idx=args.idx, n_tokens=args.tokens, out=args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
