"""Orchestrates feature extraction: generate -> capture activations -> build images -> save.

Two extraction-time choices decide WHAT gets captured and therefore WHERE it is
stored (they partition the data folder so they never collide):

  extract.source          qkv -> per-layer Q/K/V projections (forward hooks)
                          hs  -> per-layer hidden states (residual stream)
  extract.extraction_type delta      -> channels (raw, delta-prev, delta-next)
                          transforms -> channels (raw, DWT, FFT) along L

On-disk layout, per (source, extraction_type, dataset, LLM):

    data/{source}/{extraction_type}/{dataset}/{llm_alias}/
        00000/
            tokens.npy      (T, V, L, C, 3) float16
            meta.txt        human-readable prompt / response / gold / score / label
        00001/
        ...
        manifest.jsonl      one JSON line per example (the training index)
        geometry.json       the model geometry the images were built with
        progress.log        "i/total" appended every 100 generated examples

The view axis V is a REAL axis in the stored array -- it is never folded into
channels. For source=qkv that keeps "drop a view" (extract.views: [Q]) a pure
slicing operation requiring no re-extraction, and it is what lets each view get
its own CNN. For source=hs there is exactly one view (H), so V == 1.

Extraction is the expensive step (one generate() per example), so it is
restartable: an example whose directory already contains tokens.npy is skipped
unless --overwrite is passed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.extract.datasets import load_examples
from src.extract.qkv_hooks import capture_hidden, capture_qkv, read_geometry
from src.extract.tensor_ops import build_view_image, pool_layer_axis
from src.label.registry import label_examples
from src.utils.logger import get_logger
from src.utils.progress import progress

logger = get_logger(__name__)

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def load_llm(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s", cfg.llm.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm.name,
        dtype=DTYPES[cfg.llm.dtype],
        device_map="auto",
        # Q/K/V come from forward hooks on the projection Linears, so the
        # attention kernel choice does not affect what we capture.
    )
    model.eval()
    return model, tokenizer


def resolve_stop_tokens(tokenizer, model, model_name: str) -> list[int]:
    """Every id that should terminate generation.

    `tokenizer.eos_token_id` returns a SINGLE id, but instruct models stop on a
    different token than their base counterpart -- Llama-3-Instruct ends turns
    with <|eot_id|>, not <|end_of_text|>. Relying on eos_token_id alone means the
    model never stops, every response runs to max_new_tokens, and the tail of
    each image tensor is post-answer filler. So we gather every plausible stop id.
    """
    ids: set[int] = set()

    for source in (tokenizer.eos_token_id, model.config.eos_token_id):
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            ids.update(int(i) for i in source)
        else:
            ids.add(int(source))

    # Chat-template end-of-turn tokens, which are what instruct models actually
    # emit and which are frequently absent from eos_token_id.
    #
    # Note: some tokenizers raise AttributeError (rather than returning None) for
    # attributes they do not define, so every lookup here is defensive. A crash
    # in stop-token resolution would take down a multi-hour extraction run.
    try:
        unk = tokenizer.unk_token_id
    except (AttributeError, KeyError):
        unk = None
    for token in ("<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
        except (AttributeError, KeyError):
            continue
        if tid is None or tid < 0 or (unk is not None and tid == unk):
            continue
        ids.add(int(tid))

    # Base (non-instruct) models never emit a chat stop token and will happily
    # ramble into a new question, so both baselines cut them off at a newline.
    if "instruct" not in model_name.lower() and "-it" not in model_name.lower():
        try:
            newline = tokenizer.encode("\n", add_special_tokens=False)
        except (AttributeError, KeyError):
            newline = []
        if newline:
            ids.add(int(newline[-1]))

    return sorted(ids)


def build_prompt_ids(prompt: str, tokenizer, model_name: str, device) -> torch.Tensor:
    """Tokenise a prompt to a (1, prompt_len) LongTensor of input ids.

    Instruct models expect their chat template; base models take raw text.

    Both `tokenizer(...)` and `apply_chat_template(...)` may hand back either a
    bare tensor or a dict-like BatchEncoding depending on the transformers
    version (v5 changed apply_chat_template's return type), so we normalise
    rather than assuming. Getting a BatchEncoding where a tensor was expected
    fails later and confusingly, at `.ndim`.
    """
    is_instruct = "instruct" in model_name.lower() or "-it" in model_name.lower()

    if is_instruct and getattr(tokenizer, "chat_template", None):
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
    else:
        out = tokenizer(prompt, return_tensors="pt")

    # Normalise BatchEncoding / dict -> tensor.
    if not isinstance(out, torch.Tensor):
        out = out["input_ids"]

    if out.ndim == 1:
        out = out.unsqueeze(0)

    return out.to(device)


def build_images(
    qkv: dict[str, torch.Tensor],
    cfg: Config,
    n_cols: int,
) -> torch.Tensor:
    """Turn captured raw activations into the stored image tensor.

    Args:
        qkv: source=qkv -> {"Q": (T, L, D_q), "K"/"V": (T, L, D_kv)}
             source=hs  -> {"H": (T, L, D_hidden)}

    Returns:
        (T, V, L, C, 3) -- views stacked on a dedicated axis, NOT into channels.
        For source=hs, V == 1.
    """
    per_view = []
    for view in cfg.extract.views:
        img = build_view_image(
            qkv[view],
            n_cols=n_cols,
            extraction_type=cfg.extract.extraction_type,
            pool_mode=cfg.extract.pool,
            boundary_mode=cfg.extract.boundary_mode,
        )  # (T, L, C, 3)
        if cfg.extract.l_eff is not None:
            img = pool_layer_axis(img, cfg.extract.l_eff)
        per_view.append(img)

    return torch.stack(per_view, dim=1)  # (T, V, L, C, 3)


def format_meta(prompt: str, response: str, gold, score: float, label: int) -> str:
    return (
        f"prompt: {prompt}\n"
        f"response: {response}\n"
        f"gold: {gold}\n"
        f"score: {score}\n"
        f"label: {label}\n"
    )


def write_example(
    out_dir: Path,
    images: torch.Tensor,
    prompt: str,
    response: str,
    gold,
    score: float,
    label: int,
    save_dtype: torch.dtype,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "tokens.npy", images.to(save_dtype).numpy())

    (out_dir / "meta.txt").write_text(
        format_meta(prompt, response, gold, score, label), encoding="utf-8"
    )


def is_complete(out_dir: Path) -> bool:
    """True if this example has both its tensor AND a resolved label.

    An example is only safe to skip on resume when it is fully finished. See the
    call site for why the tensor's existence alone is not enough.
    """
    if not (out_dir / "tokens.npy").exists():
        return False
    meta_path = out_dir / "meta.txt"
    if not meta_path.exists():
        return False
    try:
        label = parse_meta(meta_path).get("label", "").strip()
        return label in ("0", "1")
    except OSError:
        return False


def is_generated(out_dir: Path) -> bool:
    """True if the expensive GENERATION step is already done for this example.

    That means both the image tensor and a meta.txt carrying the response/gold
    exist -- regardless of whether a label has been resolved yet. Such an example
    never needs the model again: it only needs labeling + a manifest entry, which
    run_extraction reconstructs from meta.txt rather than re-generating.
    """
    if not (out_dir / "tokens.npy").exists():
        return False
    meta_path = out_dir / "meta.txt"
    if not meta_path.exists():
        return False
    try:
        meta = parse_meta(meta_path)
    except OSError:
        return False
    # A response key must be present (it may be empty text, but the field exists
    # once generation wrote meta). gold is needed to label.
    return "response" in meta and "gold" in meta


def _record_from_meta(out_dir: Path, idx: int) -> dict:
    """Rebuild the in-memory record for an already-generated example from disk.

    Mirrors the dict appended during generation, so labeling and the manifest
    treat a reused example identically to a freshly-generated one. n_tokens is
    read from the stored tensor's first axis without loading the whole array.
    """
    meta = parse_meta(out_dir / "meta.txt")
    # mmap so we read only the header, not the full tensor, for the token count.
    n_tokens = int(np.load(out_dir / "tokens.npy", mmap_mode="r").shape[0])
    return {
        "idx": idx,
        "dir": out_dir.name,
        "n_tokens": n_tokens,
        "prompt": meta.get("prompt", ""),
        "response": meta.get("response", ""),
        "gold": meta.get("gold", ""),
    }


def run_extraction(cfg: Config, chunk: int | None = None, overwrite: bool = False) -> None:
    root = cfg.example_dir()
    root.mkdir(parents=True, exist_ok=True)

    examples = load_examples(cfg)
    logger.info("loaded %d examples for %s", len(examples), cfg.dataset.name)

    # Chunking mirrors ACT-ViT: 1-indexed blocks of 1000, so a long extraction
    # can be split across machines and resumed.
    if chunk is not None:
        lo, hi = (chunk - 1) * 1000, chunk * 1000
        examples = [e for e in examples if lo <= e.idx < hi]
        logger.info("chunk %d -> %d examples (idx %d..%d)", chunk, len(examples), lo, hi - 1)
        if not examples:
            logger.warning("chunk %d is empty; nothing to do", chunk)
            return

    # Partition the work BEFORE touching the GPU. Generation (one generate() per
    # example) is the only step that needs the LLM; labeling + manifest do not.
    #   complete    -> already labeled; skip entirely.
    #   generated   -> tensor + response on disk but unlabeled; REUSE it (no LLM),
    #                  rebuild its record from meta.txt, and let labeling finish it.
    #   to_generate -> needs the model.
    # This is what lets a run whose generation finished but crashed before
    # labeling pick up straight at the post-generation step, without re-running
    # the model on 10k prompts.
    reused: list[dict] = []
    to_generate = []
    skipped = 0
    for ex in examples:
        out_dir = root / f"{ex.idx:05d}"
        if not overwrite and is_complete(out_dir):
            skipped += 1
        elif not overwrite and is_generated(out_dir):
            reused.append(_record_from_meta(out_dir, ex.idx))
        else:
            to_generate.append(ex)

    if skipped:
        logger.info("skipped %d already-complete examples (use --overwrite to redo)", skipped)
    if reused:
        logger.info(
            "reusing %d already-generated (but unlabeled) examples: no re-generation",
            len(reused),
        )

    records: list[dict] = list(reused)

    if to_generate:
        model, tokenizer = load_llm(cfg)
        geom = read_geometry(model)
        device = next(model.parameters()).device
        logger.info("model geometry: %s", geom)

        stop_ids = resolve_stop_tokens(tokenizer, model, cfg.llm.name)
        logger.info(
            "stop tokens: %s (%s)",
            stop_ids,
            [tokenizer.decode([i]) for i in stop_ids],
        )
        if not stop_ids:
            logger.warning(
                "no stop tokens found: every response will run to max_new_tokens=%d",
                cfg.dataset.max_new_tokens,
            )

        # n_cols defaults to the layer count, which makes the image SQUARE -- that
        # is the design: we pool D down to L rather than reshaping a rectangle. Not
        # every model admits that default (Qwen2.5-7B's L=28 does not divide its
        # D_kv=512), so this raises with the valid alternatives if it cannot.
        n_cols = cfg.extract.n_cols or geom.n_layers
        geom.check_n_cols(n_cols, views=tuple(cfg.extract.views))
        for view in cfg.extract.views:
            d = geom.feature_dim(view)
            logger.info("view %s: D=%d -> %d cols (chunk width %d)", view, d, n_cols, d // n_cols)

        n_rows = cfg.extract.l_eff or geom.n_layers
        logger.info(
            "image shape per token: %d views x (%d layers x %d cols x 3 chans)",
            len(cfg.extract.views), n_rows, n_cols,
        )

        (root / "geometry.json").write_text(
            json.dumps(
                {
                    "llm": cfg.llm.name,
                    "geometry": asdict(geom),
                    "source": cfg.extract.source,
                    "extraction_type": cfg.extract.extraction_type,
                    "n_cols": n_cols,
                    "n_rows": n_rows,
                    "views": cfg.extract.views,
                    "pool": cfg.extract.pool,
                    "boundary_mode": cfg.extract.boundary_mode,
                },
                indent=2,
            )
        )

        save_dtype = DTYPES[cfg.extract.dtype]

        # Plain-text progress log next to geometry.json: one line every 100
        # examples ("i/total"). tqdm (via progress()) is a no-op under --slurm
        # (see src/utils/progress.py), so on a cluster this is otherwise the
        # only way to see how far a long extraction has gotten without
        # tailing a log file full of generation internals.
        progress_log = (root / "progress.log").open("a")
        total_to_generate = len(to_generate)

        for i, ex in enumerate(
            progress(to_generate, desc=f"extract {cfg.dataset.name}/{cfg.llm.alias}", ncols=100),
            start=1,
        ):
            out_dir = root / f"{ex.idx:05d}"
            input_ids = build_prompt_ids(ex.prompt, tokenizer, cfg.llm.name, device)

            if cfg.extract.source == "hs":
                qkv, gen_ids = capture_hidden(
                    model,
                    input_ids,
                    max_new_tokens=cfg.dataset.max_new_tokens,
                    eos_token_id=stop_ids,
                )
            else:
                qkv, gen_ids = capture_qkv(
                    model,
                    input_ids,
                    views=tuple(cfg.extract.views),
                    max_new_tokens=cfg.dataset.max_new_tokens,
                    eos_token_id=stop_ids,
                )

            if gen_ids.numel() == 0:
                logger.warning("example %d generated nothing; skipping", ex.idx)
                continue

            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Truncate to max_tokens BEFORE building images: this caps both compute
            # and disk. The response text is left whole so the label reflects what
            # the model actually said.
            if cfg.extract.max_tokens and qkv[cfg.extract.views[0]].shape[0] > cfg.extract.max_tokens:
                qkv = {v: t[: cfg.extract.max_tokens] for v, t in qkv.items()}

            images = build_images(qkv, cfg, n_cols=n_cols)

            records.append(
                {
                    "idx": ex.idx,
                    "dir": out_dir.name,
                    "n_tokens": int(images.shape[0]),
                    "prompt": ex.prompt,
                    "response": response,
                    "gold": ex.gold,
                }
            )
            # Written after labeling below; stash the tensor path for now.
            write_example(
                out_dir, images, ex.prompt, response, ex.gold,
                score=float("nan"), label=-1, save_dtype=save_dtype,
            )

            if i % 100 == 0 or i == total_to_generate:
                progress_log.write(f"{i}/{total_to_generate}\n")
                progress_log.flush()

        progress_log.close()
    elif reused:
        logger.info("nothing to generate; going straight to labeling + manifest")

    if not records:
        logger.warning("no new examples extracted")
        return

    # ---- label, and rewrite meta with the resolved score/label -------------
    logger.info("labeling %d examples with scheme=%s", len(records), cfg.labeling.scheme)
    scored = label_examples(
        cfg, [r["response"] for r in records], [r["gold"] for r in records]
    )

    for rec, (score, label) in zip(records, scored):
        rec["score"] = score
        rec["label"] = label
        out_dir = root / rec["dir"]
        (out_dir / "meta.txt").write_text(
            format_meta(rec["prompt"], rec["response"], rec["gold"], score, label),
            encoding="utf-8",
        )

    write_manifest(root, records, chunk)

    n_hall = sum(r["label"] for r in records)
    logger.info(
        "done: %d examples, %d hallucinated (%.1f%%)",
        len(records), n_hall, 100 * n_hall / len(records),
    )
    if n_hall == 0 or n_hall == len(records):
        logger.warning(
            "DEGENERATE LABELS: every example has the same label. The classifier "
            "cannot learn anything. Check the prompt template and the gold field."
        )
    elif min(n_hall, len(records) - n_hall) / len(records) < 0.05:
        logger.warning(
            "labels are highly imbalanced (%.1f%% minority). AUROC will be noisy; "
            "consider a harder dataset.", 100 * min(n_hall, len(records) - n_hall) / len(records),
        )


def write_manifest(root: Path, new_records: list[dict], chunk: int | None) -> None:
    """Merge new records into manifest.jsonl, keyed by idx (last write wins).

    Chunked runs each append their own slice, so we re-read and merge rather than
    truncating -- otherwise chunk 2 would erase chunk 1's entries.
    """
    path = root / "manifest.jsonl"
    merged: dict[int, dict] = {}

    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    merged[rec["idx"]] = rec

    for rec in new_records:
        slim = {k: rec[k] for k in ("idx", "dir", "n_tokens", "score", "label")}
        merged[rec["idx"]] = slim

    with open(path, "w") as f:
        for idx in sorted(merged):
            f.write(json.dumps(merged[idx]) + "\n")

    logger.info("manifest now has %d examples: %s", len(merged), path)


def relabel(cfg: Config) -> None:
    """Recompute labels from the stored responses WITHOUT re-extracting features.

    This is why meta.txt keeps the response and gold: swapping exact_match for
    BLEURT is a cheap CPU pass, not a multi-hour GPU re-run.
    """
    root = cfg.example_dir()
    path = root / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}; run `extract` first")

    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    responses, golds = [], []
    for rec in records:
        meta = parse_meta(root / rec["dir"] / "meta.txt")
        responses.append(meta["response"])
        golds.append(meta["gold"])

    logger.info("relabeling %d examples with scheme=%s", len(records), cfg.labeling.scheme)
    scored = label_examples(cfg, responses, golds)

    for rec, (score, label) in zip(records, scored):
        rec["score"], rec["label"] = score, label

    with open(path, "w") as f:
        for rec in sorted(records, key=lambda r: r["idx"]):
            f.write(json.dumps(rec) + "\n")

    n_hall = sum(r["label"] for r in records)
    logger.info("relabeled: %d hallucinated of %d (%.1f%%)",
                n_hall, len(records), 100 * n_hall / len(records))


def parse_meta(path: Path) -> dict:
    """Parse meta.txt. Fields are single-line 'key: value'; the gold field may be
    a stringified list, which the labelers handle."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        for key in ("prompt", "response", "gold", "score", "label"):
            prefix = f"{key}: "
            if line.startswith(prefix):
                out[key] = line[len(prefix):]
                current = key
                break
        else:
            # Continuation of a multi-line field (a response with newlines).
            if current:
                out[current] += "\n" + line
    return out
