#!/usr/bin/env python3
"""
Train V1~V5 → CSV 추출
=======================
train_1000.json에서 V1~V5 샘플을 레이블별 CSV로 분리.
Google Sheets에서 수정 후 merge_csv_to_json.py로 재조립.

CSV 컬럼 (27개):
  session_id, label, situation,
  prefix1~prefix20  (참고용, readonly),
  gen1, gen2, gen3, gen4,  (수정 대상)
  fixed  (수정 완료 시 TRUE)

- prefix: "[seeker] 내용" 또는 "[supporter] 내용" 형태 (참고만)
- gen1~gen4: content만 (speaker는 고정: seeker/supporter/seeker/supporter)
- 12턴 미만 prefix는 빈 칸

사용법:
    python3 extract_train_csv.py

출력:
    csv_edit/train_V1_edit.csv
    csv_edit/train_V2_edit.csv
    csv_edit/train_V3_edit.csv
    csv_edit/train_V4_edit.csv
    csv_edit/train_V5_edit.csv
"""

import json
import csv
import os

INPUT_PATH = "data/final/train_1000.json"
OUTPUT_DIR = "csv_edit"
MAX_PREFIX = 20
LABELS_TO_EXTRACT = ["Normal", "V1", "V2", "V3", "V4", "V5"]

# gen turn speakers (고정 패턴)
# gen1=seeker, gen2=supporter, gen3=seeker, gen4=supporter(TARGET)

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

    # gen1~gen4 (content만, speaker는 위치로 결정)
    gen_cells = [t["content"].strip().replace("\n", " ") for t in gen]

    # 이미 수정된 샘플은 fixed=TRUE로 표시
    fixed = "TRUE" if sample.get("_edited") else "FALSE"
    
    row = (
        [sample["esconv_session_id"], sample["primary_label"],
         sample["situation"].strip().replace("\n", " ")]
        + prefix_cells
        + gen_cells
        + [fixed]
    )
    return row


def main():
    # Load
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 레이블별 분리 & CSV 저장
    total = 0
    for label in LABELS_TO_EXTRACT:
        label_samples = [s for s in samples if s["primary_label"] == label]
        if not label_samples:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"train_{label}_edit.csv")
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(HEADER)
            for s in label_samples:
                writer.writerow(sample_to_row(s))

        print(f"  {label}: {len(label_samples):3d} samples → {out_path}")
        total += len(label_samples)

    print(f"\n  Total V1~V5: {total} samples (Normal {len(samples)-total}개 제외)")
    print(f"  출력 디렉토리: {OUTPUT_DIR}/")
    print(f"\n  ※ gen1~gen4 수정 후 fixed 컬럼을 TRUE로 변경")
    print(f"  ※ 완료 후: python3 merge_csv_to_json.py")


if __name__ == "__main__":
    main()
