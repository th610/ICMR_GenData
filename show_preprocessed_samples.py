"""Show preprocessed data samples."""
import json

def show_sample(filepath, class_name):
    data = json.load(open(filepath, encoding='utf-8'))
    print(f"\n{'='*60}")
    print(f"=== {class_name} 샘플 ===")
    print(f"{'='*60}")
    
    s = data[0]
    print(f"\nSession ID: {s['session_id']}")
    print(f"Label: {s['label']}")
    
    print(f"\n[SUMMARY]")
    print(s['summary'][0])
    
    print(f"\n[CONTEXT - {len(s['context_turns'])}턴]")
    for i, t in enumerate(s['context_turns'], 1):
        speaker = t['speaker']
        text = t['text']
        if len(text) > 100:
            text = text[:100] + "..."
        print(f"{i}. {speaker}: {text}")
    
    print(f"\n[RESPONSE]")
    resp = s['response']
    if len(resp) > 200:
        resp = resp[:200] + "..."
    print(resp)

# Show samples
show_sample('data/processed/normal_processed.json', 'NORMAL')
show_sample('data/processed/v1_processed.json', 'V1 (Empathy Violation)')
show_sample('data/processed/v2_processed.json', 'V2 (Fact-Checking Violation)')
show_sample('data/processed/v3_processed.json', 'V3 (Advice Violation)')
show_sample('data/processed/v4_processed.json', 'V4 (Multiturn Mixed)')
show_sample('data/processed/v5_processed.json', 'V5 (Multiturn Single)')

print(f"\n{'='*60}")
print("전처리 형식:")
print("  - 요약: LLM이 전체 대화 요약 (1개 bullet)")
print("  - 컨텍스트: 마지막 5-6턴 윈도우")
print("  - 응답: 마지막 supporter 발화")
print("  - 평균 토큰: 174 (max 419, truncation 0%)")
print(f"{'='*60}\n")
