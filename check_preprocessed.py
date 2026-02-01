"""Check preprocessed data summary."""
import json
import os

files = {
    'normal': 'data/processed/normal_processed.json',
    'v1': 'data/processed/v1_processed.json',
    'v2': 'data/processed/v2_processed.json',
    'v3': 'data/processed/v3_processed.json',
    'v4': 'data/processed/v4_processed.json',
    'v5': 'data/processed/v5_processed.json',
}

print("\n=== 전처리 데이터 현황 ===\n")

total = 0
for cls, path in files.items():
    if os.path.exists(path):
        data = json.load(open(path, encoding='utf-8'))
        print(f"{cls.upper()}: {len(data)}개")
        total += len(data)
        
        # Show sample format
        if len(data) > 0:
            sample = data[0]
            print(f"  - 필드: {list(sample.keys())}")
            if 'summary' in sample:
                print(f"  - 요약 포인트: {len(sample['summary'])}개")
            if 'context_turns' in sample:
                print(f"  - 컨텍스트 턴: {len(sample['context_turns'])}개")
            print()

print(f"총 {total}개 전처리 완료")
print(f"\n형식: 요약(bullets) + 컨텍스트(window=6) + 응답")
