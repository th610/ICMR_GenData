#!/usr/bin/env python3
"""
Gold CSV → JSON 재조립
=======================
수정된 gold CSV를 읽어서 test_gold_300.json에 병합.

로직:
  1. 원본 test_gold_300.json 로드
  2. csv_edit_gold/gold_*_edit.csv 읽기
  3. fixed == TRUE인 샘플만 gen1~gen4 교체
  4. session_id 기준으로 매칭

사용법:
    python3 merge_gold_csv_to_json.py               # 기본: 덮어쓰기
    python3 merge_gold_csv_to_json.py --dry-run      # 미리보기
    python3 merge_gold_csv_to_json.py --out new.json  # 다른 파일
"""

import json
import csv
import os
import shutil
import datetime
import argparse
from collections import defaultdict

INPUT_JSON = "data/final/test_gold_300.json"
CSV_DIR = "csv_edit_gold"
LABELS = ["Normal", "V1", "V2", "V3", "V4", "V5"]
GEN_SPEAKERS = ["seeker", "supporter", "seeker", "supporter"]


def load_csv_edits():
    """csv_edit_gold/ 에서 fixed=TRUE인 수정사항 로드."""
    edits = {}  # session_id → {gen1, gen2, gen3, gen4, label}

    for label in LABELS:
        csv_path = os.path.join(CSV_DIR, f"gold_{label}_edit.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠  {csv_path} 없음 → 건너뜀")
            continue

        count = 0
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["session_id"])
                fixed = row.get("fixed", "").strip().upper()

                if sid in edits and fixed != "TRUE":
                    continue

                if fixed == "TRUE":
                    edits[sid] = {
                        "gen1": row["gen1"].strip(),
                        "gen2": row["gen2"].strip(),
                        "gen3": row["gen3"].strip(),
                        "gen4": row["gen4"].strip(),
                        "label": row["label"].strip(),
                    }
                    count += 1

        print(f"  {label}: {count} edits loaded from {csv_path}")

    return edits


def apply_edits(data, edits):
    """원본 JSON에 수정사항 적용"""
    samples = data["samples"]

    sid_map = {}
    for i, s in enumerate(samples):
        sid = s["esconv_session_id"]
        sid_map[sid] = i

    applied = 0
    not_found = []
    label_counts = defaultdict(int)

    for sid, edit in edits.items():
        if sid not in sid_map:
            not_found.append(sid)
            continue

        idx = sid_map[sid]
        sample = samples[idx]

        for gi, gen_key in enumerate(["gen1", "gen2", "gen3", "gen4"]):
            new_content = edit[gen_key]
            if new_content:
                sample["generated_dialog"][gi]["content"] = new_content
                sample["generated_dialog"][gi]["speaker"] = GEN_SPEAKERS[gi]

        # CSV에서 라벨이 변경된 경우 반영
        if edit["label"] and edit["label"] != sample.get("label"):
            sample["label"] = edit["label"]

        sample["_edited"] = True
        sample["_edit_timestamp"] = datetime.datetime.now().isoformat()

        applied += 1
        label_counts[edit["label"]] += 1

    print(f"\n  적용: {applied}개")
    for label in LABELS:
        if label_counts.get(label):
            print(f"    {label}: {label_counts[label]}개")

    if not_found:
        print(f"  ⚠  session_id 매칭 실패: {not_found}")

    data["metadata"]["edited_count"] = applied
    data["metadata"]["edit_timestamp"] = datetime.datetime.now().isoformat()
    data["metadata"]["edit_distribution"] = dict(label_counts)

    return data, applied


def main():
    parser = argparse.ArgumentParser(description="Gold CSV → JSON 재조립")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    output_path = args.out or INPUT_JSON

    print("=" * 60)
    print("Gold CSV → JSON 재조립")
    print("=" * 60)

    print(f"\n  원본: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  샘플 수: {len(data['samples'])}")

    print(f"\n  CSV 디렉토리: {CSV_DIR}/")
    edits = load_csv_edits()
    print(f"  총 수정: {len(edits)}개")

    if not edits:
        print("\n  ⚠  수정된 항목 없음. 종료.")
        return

    data, applied = apply_edits(data, edits)

    if args.dry_run:
        print(f"\n  [DRY RUN] 저장하지 않음.")
        return

    if output_path == INPUT_JSON:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = INPUT_JSON + f".bak_{ts}"
        shutil.copy2(INPUT_JSON, backup_path)
        print(f"\n  백업: {backup_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  저장: {output_path}")
    print(f"  완료: {applied}개 수정 반영")


if __name__ == "__main__":
    main()
