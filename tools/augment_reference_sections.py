#!/usr/bin/env python3
"""
Augment reference ground-truth files with form and section from CRF pages.

Inputs:
- reference_dir: directory containing page_XXX_result.json files
- crf_dir: directory containing page_XXX.json files (from blankCRF_corrected)

Output:
- Writes augmented files to output_dir mirroring filenames, adding
  'form' and 'section' to each annotation via string matching to <Q> items.

Matching strategy:
- For each reference annotation.text, find best <Q> by difflib ratio.
- Use <Q>.form and <Q>.section; if section is missing, infer by choosing the
  nearest section header above the question using bbox Y coordinate.
- Page-level form fallback: use <FORM> tag or parse "Form: ..." supplemental.

Usage:
  python tools/augment_reference_sections.py \
    --reference-dir reference \
    --crf-dir crf_json/blankCRF_corrected \
    --output-dir reference_with_sections
"""

import argparse
import json
from pathlib import Path
from difflib import SequenceMatcher
import re


def norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_crf_page(crf_path: Path):
    data = json.loads(crf_path.read_text())
    items = data.get("items", [])
    supp = data.get("supplemental_page_data", [])

    # Extract questions
    questions = []
    for it in items:
        if it.get("tag") == "<Q>":
            bbox = it.get("bbox_xyxy") or [0, 0, 0, 0]
            questions.append({
                "text": it.get("text", ""),
                "text_norm": norm_text(it.get("text", "")),
                "form": it.get("form"),
                "section": it.get("section"),
                "y": float(bbox[1]) if isinstance(bbox, list) and len(bbox) >= 2 else 0.0,
            })

    # Extract section headers with approximate Y
    sections = []
    for sp in supp:
        if sp.get("type") == "section_header":
            bbox = sp.get("bbox") or [0, 0, 0, 0]
            sections.append({
                "text": sp.get("text", ""),
                "text_norm": norm_text(sp.get("text", "")),
                "y": float(bbox[1]) if isinstance(bbox, list) and len(bbox) >= 2 else 0.0,
            })
    sections.sort(key=lambda x: x["y"])  # top to bottom

    # Page-level form
    page_form = None
    for it in items:
        if it.get("tag") == "<FORM>":
            page_form = it.get("text")
            break
    if not page_form:
        for sp in supp:
            if sp.get("type") == "supplemental":
                m = re.search(r"Form:\s*([^\n]+)", sp.get("text", ""))
                if m:
                    page_form = m.group(1).strip()
                    break

    # Fill missing sections by nearest header above question
    for q in questions:
        if not q.get("section") and sections:
            # pick section with y <= q.y and largest y
            above = [sec for sec in sections if sec["y"] <= q["y"]]
            if above:
                q["section"] = above[-1]["text"]

    return questions, page_form


def find_best_question(ann_text: str, questions: list):
    target = norm_text(ann_text)
    best = None
    best_score = 0.0
    for q in questions:
        # Primary: direct similarity of full text
        score = SequenceMatcher(None, target, q["text_norm"]).ratio()
        # Light bonus if annotation text prefix matches question prefix
        if q["text_norm"].startswith(target[:30]):
            score += 0.05
        if score > best_score:
            best_score = score
            best = q
    return best, best_score


def augment_file(ref_path: Path, crf_dir: Path, out_dir: Path):
    # Determine page filename
    m = re.search(r"page_(\d+)_result\.json$", ref_path.name)
    if not m:
        print(f"Skip (not page_*_result.json): {ref_path.name}")
        return
    page_idx = int(m.group(1))
    crf_path = crf_dir / f"page_{page_idx:03d}.json"
    if not crf_path.exists():
        print(f"CRF not found for {ref_path.name}: {crf_path}")
        return

    questions, page_form = load_crf_page(crf_path)

    ref = json.loads(ref_path.read_text())
    anns = ref.get("annotations", [])

    augmented = 0
    for ann in anns:
        qtext = ann.get("text", "")
        match, score = find_best_question(qtext, questions)
        if match:
            # Always take the most similar question on the page
            ann["form"] = match.get("form") or page_form
            ann["section"] = match.get("section")
            augmented += 1
        else:
            # Fallback when page has no <Q> items
            ann["form"] = page_form
            ann["section"] = None

    ref["annotations"] = anns
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ref_path.name
    out_path.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print(f"Augmented {ref_path.name}: {augmented}/{len(anns)} annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", default="reference")
    ap.add_argument("--crf-dir", default="crf_json/blankCRF_corrected")
    ap.add_argument("--output-dir", default="reference_with_sections")
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    crf_dir = Path(args.crf_dir)
    out_dir = Path(args.output_dir)

    files = sorted(ref_dir.glob("page_*_result.json"))
    if not files:
        print(f"No reference files in {ref_dir}")
        return

    for f in files:
        augment_file(f, crf_dir, out_dir)


if __name__ == "__main__":
    main()
