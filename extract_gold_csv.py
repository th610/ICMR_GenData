#!/usr/bin/env python3
"""
Gold Test Set (300개) → CSV 추출
================================
test_gold_300.json에서 레이블별 CSV로 분리.
Google Sheets에서 수정 후 merge_csv_to_json.py로 재조립.

CSV 컬럼 (28개):
  session_id, label, situation,
  prefix1~prefix20  (참고용, readonly),
  gen1, gen2, gen3, gen4,  (수정 대상)
  fixed  (수정 완료 시 TRUE)

사용법:
    python3 extract_gold_csv.py

출력:
    csv_edit_gold/gold_Normal_edit.csv
    csv_edit_gold/gold_V1_edit.csv
    ...
"""

import json
import csv
import os

INPUT_PATH = "data/final/test_gold_300.json"
OUTPUT_DIR = "csv_edit_gold"
MAX_PREFIX = 20
LABELS_TO_EXTRACT = ["Normal", "V1", "V2", "V3", "V4", "V5"]

HEADER = (
    ["session_id", "label", "situation"]
    + [f"prefix{i}" for i in range(1, MAX_PREFIX + 1)]
    + ["gen1", "gen2", "gen3", "gen4"]
    + ["fixed"]
)


def format_prefix_turn(turn):
    """prefix 턴을 [speaker] content 형태로"""
    speaker = turn["speaker"].lower()
    content = turn["content"].strip().replace("\n", " ")
    return f"[{speaker}] {content}"


def sample_to_row(sample):
    """JSON sample → CSV row"""
    prefix = sample.get("prefix_dialog", [])
    gen = sample["generated_dialog"]

    # prefix1~prefix20
    prefix_cells = []
    for i in range(MAX_PREFIX):
        if i < len(prefix):
            prefix_cells.append(format_prefix_turn(prefix[i]))
        else:
            prefix_cells.append("")

    # gen1~gen4 (content만)
    gen_cells = [t["content"].strip().replace("\n", " ") for t in gen]

    label = sample.get("primary_label") or sample.get("label")

    row = (
        [sample["esconv_session_id"], label,
         sample["situation"].strip().replace("\n", " ")]
        + prefix_cells
        + gen_cells
        + ["FALSE"]
    )
    return row


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    for label in LABELS_TO_EXTRACT:
        label_samples = [s for s in samples
                         if (s.get("primary_label") or s.get("label")) == label]
        if not label_samples:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"gold_{label}_edit.csv")
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(HEADER)
            for s in label_samples:
                writer.writerow(sample_to_row(s))

        print(f"  {label}: {len(label_samples):3d} samples → {out_path}")
        total += len(label_samples)

    print(f"\n  Total: {total} samples")
    print(f"  출력 디렉토리: {OUTPUT_DIR}/")
    print(f"\n  ※ gen1~gen4 수정 후 fixed 컬럼을 TRUE로 변경")


if __name__ == "__main__":
    main()
