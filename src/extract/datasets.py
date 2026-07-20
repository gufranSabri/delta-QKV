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
    """Cap context length by words. Long HotpotQA contexts would
    otherwise dominate the prompt and slow generation to a crawl."""
    words = text.split()
    return " ".join(words[:n])


def load_triviaqa(cfg, n: int, split: str = "train") -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)

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
        # `best_answer` ONLY -- HalluShift discards `correct_answers` when
        # building references (hal_detection.py:316). Including them would make
        # our labels strictly more lenient than theirs, and the AUROCs would no
        # longer be measuring the same task.
        gold = row["best_answer"]
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(question=row["question"]),
                gold=[gold] if gold else [],
                idx=len(out),
            )
        )
    return out


def load_tydiqa(cfg, n: int, split: str = "train") -> list[Example]:
    """TyDiQA-GP, English only -- HalluShift's `tydiqa` (hal_detection.py:64-65)."""
    from datasets import load_dataset

    ds = load_dataset("tydiqa", "secondary_task", split=split)
    ds = ds.filter(lambda row: "english" in row["id"])

    out = []
    for row in ds:
        if len(out) >= n:
            break
        context = _truncate_words(row["context"], 300)
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(
                    context=context, question=row["question"]
                ),
                gold=list(row["answers"]["text"]),
                idx=len(out),
            )
        )
    return out


def _coqa_rows(split: str) -> list[dict]:
    """Download CoQA and flatten each dialogue turn into its own row.

    Mirrors HalluShift (hal_detection.py:81-128): the story ACCUMULATES the
    preceding Q/A pairs, so turn k is conditioned on the dialogue so far. Getting
    this wrong would make our contexts -- and therefore the task -- different.
    """
    import json
    import urllib.request
    from pathlib import Path

    from src.config import REPO_ROOT

    fname = "coqa-train-v1.0.json" if split == "train" else "coqa-dev-v1.0.json"
    dest = Path(REPO_ROOT) / "data" / "raw" / "coqa" / fname
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://downloads.cs.stanford.edu/nlp/data/coqa/{fname}"
        urllib.request.urlretrieve(url, dest)

    rows: list[dict] = []
    for sample in json.loads(dest.read_text())["data"]:
        story = sample["story"]
        for i, question in enumerate(sample["questions"]):
            answer = sample["answers"][i]["input_text"]
            rows.append({"story": story, "question": question["input_text"], "answer": answer})
            # Append this turn to the running context for the NEXT question.
            story += f' Q: {question["input_text"]} A: {answer}'
            if story and story[-1] != ".":
                story += "."
    return rows


def load_coqa(cfg, n: int, split: str = "train") -> list[Example]:
    out = []
    for row in _coqa_rows(split):
        if len(out) >= n:
            break
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(
                    story=_truncate_words(row["story"], 300), question=row["question"]
                ),
                # HalluShift uses `answer['text']` only, discarding the three
                # `additional_answers` (hal_detection.py:323).
                gold=[row["answer"]],
                idx=len(out),
            )
        )
    return out


def load_hotpotqa(cfg, n: int, with_context: bool, split: str = "train") -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
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


LOADERS = {
    "triviaqa": lambda cfg, n, sp: load_triviaqa(cfg, n, split=sp),
    "truthfulqa": lambda cfg, n, sp: load_truthfulqa(cfg, n),
    "hotpotqa": lambda cfg, n, sp: load_hotpotqa(cfg, n, with_context=False, split=sp),
    "hotpotqa_with_context": lambda cfg, n, sp: load_hotpotqa(cfg, n, with_context=True, split=sp),
    "coqa": lambda cfg, n, sp: load_coqa(cfg, n, split=sp),
    "tydiqa": lambda cfg, n, sp: load_tydiqa(cfg, n, split=sp),
}

# Which upstream split backs the train corpus vs. the held-out `<name>_test` twin.
#
# This mapping is the whole point: ACT-ViT trains on the benchmark's train split
# and evaluates on a *separately generated* corpus built from the benchmark's
# dev/test split (ACT-ViT/utils/datasets_helper.py:281-324). Reproducing their
# numbers means reproducing that separation. HotpotQA and TriviaQA have no public
# labelled test split, so their dev set plays the test role -- same as ACT-ViT.
SPLIT_SOURCES = {
    # NOTE on triviaqa/hotpotqa: their upstream `test` splits are UNLABELLED
    # leaderboard blind sets, so `validation` is the real held-out set and plays
    # the test role. Same choice ACT-ViT and HalluShift make.
    "triviaqa": {"train": "train", "test": "validation"},
    "hotpotqa": {"train": "train", "test": "validation"},
    "hotpotqa_with_context": {"train": "train", "test": "validation"},
    "coqa": {"train": "train", "test": "dev"},
    "tydiqa": {"train": "train", "test": "validation"},
    # TruthfulQA has exactly one split (817 rows, `validation`) and no held-out
    # set to speak of, so there is no `truthfulqa_test`. It instead gets a
    # stratified test slice carved out at train time -- see make_split().
    "truthfulqa": {"train": "validation"},
}

# Datasets with a genuine held-out corpus. `test.py` prefers `<name>_test` for
# these and falls back to the in-split holdout for anything else.
HAS_TEST_CORPUS = {n for n, s in SPLIT_SOURCES.items() if "test" in s}


def load_examples(cfg) -> list[Example]:
    name, is_test = base_name(cfg.dataset.name)
    if name not in LOADERS:
        raise KeyError(
            f"no loader for dataset {cfg.dataset.name!r}. Known: {sorted(LOADERS)}"
        )
    sources = SPLIT_SOURCES[name]
    if is_test and "test" not in sources:
        raise KeyError(
            f"{name!r} has no held-out test corpus (it has a single upstream split). "
            f"Evaluate on {name!r} directly; test.py will use its held-out slice."
        )
    split = sources["test" if is_test else "train"]
    return LOADERS[name](cfg, cfg.dataset.n_samples, split)


def base_name(dataset_name: str) -> tuple[str, bool]:
    """Split `foo_test` into ("foo", True); anything else into (name, False)."""
    if dataset_name.endswith("_test"):
        return dataset_name[: -len("_test")], True
    return dataset_name, False
