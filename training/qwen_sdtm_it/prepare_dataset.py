#!/usr/bin/env python3
"""
Prepare instruction-tuning data for Qwen2.5-14B from reference_with_sections.

Creates chat-style JSONL files with messages: [system, user, assistant].
"""
import argparse
import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an SDTM mapping assistant. Given a CRF question, form and section, "
    "produce the correct SDTM annotation exactly as requested. Use the canonical "
    "format found in the training examples."
)


def build_message(form: str, section: str, question: str, target: str):
    user_lines = []
    if form:
        user_lines.append(f"Form: {form}")
    if section:
        user_lines.append(f"Section: {section}")
    user_lines.append(f"Question: {question}")
    user_lines.append("Return SDTM annotation:")
    user = "\n".join(user_lines)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ]
    }


def iter_reference(reference_dir: Path):
    for path in sorted(reference_dir.glob("page_*_result.json")):
        data = json.loads(path.read_text())
        anns = data.get("annotations", [])
        for ann in anns:
            q = (ann.get("text") or "").strip()
            y = (ann.get("annotation") or "").strip()
            if not q or not y:
                continue
            form = (ann.get("form") or "").strip()
            section = (ann.get("section") or "").strip()
            yield build_message(form, section, q, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_reference(ref_dir))
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if rows else 0

    train = rows[n_val:]
    val = rows[:n_val]

    (out_dir / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train))
    (out_dir / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val))

    print(f"Prepared {len(train)} train and {len(val)} val samples in {out_dir}")


if __name__ == "__main__":
    main()

