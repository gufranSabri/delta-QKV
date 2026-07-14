"""Text dataset loading: prompt construction and gold-answer extraction.

Each loader returns a list of Example(prompt, gold), where `gold` is whatever the
labeling scheme needs to judge correctness (a string, or a list of acceptable
aliases). Prompt templates follow HalluShift/ACT-ViT so responses are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    prompt: str
    gold: object   # str, or list[str] of acceptable answers
    idx: int


def _truncate_words(text: str, n: int) -> str:
    """Cap context length by words. Long IMDB reviews / HotpotQA contexts would
    otherwise dominate the prompt and slow generation to a crawl."""
    words = text.split()
    return " ".join(words[:n])


def load_triviaqa(cfg, n: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    # Deduplicate by question_id -- TriviaQA repeats questions across contexts,
    # and HalluShift drops the duplicates before generating.
    seen: set = set()
    out: list[Example] = []
    for row in ds:
        if len(out) >= n:
            break
        qid = row["question_id"]
        if qid in seen:
            continue
        seen.add(qid)
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(question=row["question"]),
                # All aliases count as correct -- "JFK" and "John F. Kennedy".
                gold=list(row["answer"]["aliases"]) or [row["answer"]["value"]],
                idx=len(out),
            )
        )
    return out


def load_truthfulqa(cfg, n: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for row in ds:
        if len(out) >= n:
            break
        golds = [row["best_answer"], *row.get("correct_answers", [])]
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(question=row["question"]),
                gold=[g for g in golds if g],
                idx=len(out),
            )
        )
    return out


def load_hotpotqa(cfg, n: int, with_context: bool) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    out = []
    for row in ds:
        if len(out) >= n:
            break
        if with_context:
            # Flatten the supporting paragraphs into one context block.
            paras = ["".join(sents) for sents in row["context"]["sentences"]]
            context = _truncate_words(" ".join(paras), 300)
            prompt = cfg.dataset.prompt_template.format(
                context=context, question=row["question"]
            )
        else:
            prompt = cfg.dataset.prompt_template.format(question=row["question"])
        out.append(Example(prompt=prompt, gold=row["answer"], idx=len(out)))
    return out


def load_imdb(cfg, n: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("imdb", split="test").shuffle(seed=42)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        review = _truncate_words(row["text"], 200)
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(review=review),
                gold=int(row["label"]),   # 0 = negative, 1 = positive
                idx=len(out),
            )
        )
    return out


def load_movies(cfg, n: int) -> list[Example]:
    """ACT-ViT ships this one as a CSV rather than an HF dataset."""
    import csv
    from pathlib import Path

    from src.config import REPO_ROOT

    path = Path(REPO_ROOT) / "ACT-ViT" / "data" / "movie_qa_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"movies dataset CSV not found at {path}. It ships with the ACT-ViT repo."
        )

    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if len(out) >= n:
                break
            question = row.get("question") or row.get("Question")
            answer = row.get("answer") or row.get("Answer")
            if not question or not answer:
                continue
            out.append(
                Example(
                    prompt=cfg.dataset.prompt_template.format(question=question),
                    gold=answer,
                    idx=len(out),
                )
            )
    return out


LOADERS = {
    "triviaqa": lambda cfg, n: load_triviaqa(cfg, n),
    "truthfulqa": lambda cfg, n: load_truthfulqa(cfg, n),
    "hotpotqa": lambda cfg, n: load_hotpotqa(cfg, n, with_context=False),
    "hotpotqa_with_context": lambda cfg, n: load_hotpotqa(cfg, n, with_context=True),
    "imdb": lambda cfg, n: load_imdb(cfg, n),
    "movies": lambda cfg, n: load_movies(cfg, n),
}


def load_examples(cfg) -> list[Example]:
    name = cfg.dataset.name.replace("_test", "")
    if name not in LOADERS:
        raise KeyError(
            f"no loader for dataset {cfg.dataset.name!r}. Known: {sorted(LOADERS)}"
        )
    return LOADERS[name](cfg, cfg.dataset.n_samples)
