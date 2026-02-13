#!/usr/bin/env python3
"""
CSV → JSON 재조립
==================
수정된 CSV를 읽어서 train_1000.json에 병합.

로직:
  1. 원본 train_1000.json 로드
  2. csv_edit/train_V*_edit.csv 읽기
  3. fixed == TRUE인 샘플만 gen1~gen4 교체
  4. session_id 기준으로 매칭 (index X)
  5. 원본 백업 후 새 JSON 저장

gen turn speaker (고정):
  gen1 = seeker
  gen2 = supporter
  gen3 = seeker
  gen4 = supporter (TARGET)

사용법:
    python3 merge_csv_to_json.py                    # 기본: train_1000.json 덮어쓰기
    python3 merge_csv_to_json.py --dry-run          # 미리보기만 (저장 안 함)
    python3 merge_csv_to_json.py --out new.json     # 다른 파일로 저장
"""

import json
import csv
import os
import sys
import shutil
import datetime
import argparse
from collections import defaultdict

INPUT_JSON = "data/final/train_1000.json"
CSV_DIR = "csv_edit"  # 수정된 CSV 폴더
CSV_PATTERN = "train_{label}_edit.csv"  # 파일명 패턴
LABELS = ["Normal", "V1", "V2", "V3", "V4", "V5"]

# gen turn speakers (고정)
GEN_SPEAKERS = ["seeker", "supporter", "seeker", "supporter"]


def load_csv_edits():
    """csv_edit/ 에서 fixed=TRUE인 수정사항 로드. session_id → gen1~gen4
    
    ⚠ 중요: situation, prefix는 CSV에서 읽지 않음 (원본 유지)
    중복 session_id가 있으면 fixed=TRUE 우선
    """
    edits = {}  # session_id → {gen1, gen2, gen3, gen4, label}

    for label in LABELS:
        csv_path = os.path.join(CSV_DIR, CSV_PATTERN.format(label=label))
        if not os.path.exists(csv_path):
            print(f"  ⚠  {csv_path} 없음 → 건너뜀")
            continue

        count = 0
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["session_id"])
                fixed = row.get("fixed", "").strip().upper()
                
                # 중복 시: fixed=TRUE 우선
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


def apply_edits(data, edits, dry_run=False):
    """원본 JSON에 수정사항 적용"""
    samples = data["samples"]

    # session_id → index 매핑
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

        # gen1~gen4 교체
        for gi, gen_key in enumerate(["gen1", "gen2", "gen3", "gen4"]):
            new_content = edit[gen_key]
            if new_content:  # 비어있으면 원본 유지
                sample["generated_dialog"][gi]["content"] = new_content
                sample["generated_dialog"][gi]["speaker"] = GEN_SPEAKERS[gi]

        # 메타데이터 추가
        sample["_edited"] = True
        sample["_edit_timestamp"] = datetime.datetime.now().isoformat()

        applied += 1
        label_counts[edit["label"]] += 1

    print(f"\n  적용: {applied}개")
    for label in LABELS:
        if label_counts[label]:
            print(f"    {label}: {label_counts[label]}개")

    if not_found:
        print(f"  ⚠  session_id 매칭 실패: {not_found}")

    # metadata 업데이트
    data["metadata"]["edited_count"] = applied
    data["metadata"]["edit_timestamp"] = datetime.datetime.now().isoformat()
    data["metadata"]["edit_distribution"] = dict(label_counts)

    return data, applied


def main():
    parser = argparse.ArgumentParser(description="CSV → JSON 재조립")
    parser.add_argument("--dry-run", action="store_true", help="미리보기만 (저장 안 함)")
    parser.add_argument("--out", default=None, help="출력 JSON 경로 (기본: 원본 덮어쓰기)")
    args = parser.parse_args()

    output_path = args.out or INPUT_JSON

    print("=" * 60)
    print("CSV → JSON 재조립")
    print("=" * 60)

    # 1. 원본 로드
    print(f"\n  원본: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  샘플 수: {len(data['samples'])}")

    # 2. CSV edits 로드
    print(f"\n  CSV 디렉토리: {CSV_DIR}/")
    edits = load_csv_edits()
    print(f"  총 수정: {len(edits)}개")

    if not edits:
        print("\n  ⚠  수정된 항목 없음 (fixed=TRUE인 행이 없음). 종료.")
        return

    # 3. 적용
    data, applied = apply_edits(data, edits, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n  [DRY RUN] 저장하지 않음. 실제 적용 시 --dry-run 제거.")
        return

    # 4. 백업 & 저장
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
