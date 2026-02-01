import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# 원본 데이터 샘플 확인
print("=" * 60)
print("원본 데이터 토큰 길이 확인 (전체 concat 방식)")
print("=" * 60)

file_mapping = {
    'normal': 'data/generated/normal_400.json',
    'v1': 'data/generated/v1_240.json',
    'v2': 'data/generated/v2_160.json',
    'v3': 'data/generated/v3_200.json',
    'v4': 'data/pilot/v4_full_150.json',
    'v5': 'data/pilot/v5_full_150.json'
}

for f in file_mapping.keys():
    data = json.load(open(file_mapping[f], encoding='utf-8'))
    
    # 첫 3개 샘플 토큰 길이 확인
    lengths = []
    for i, session in enumerate(data[:3]):
        # session_to_text 로직 재현
        dialog = session.get('dialog', [])
        text_parts = []
        for turn in dialog:
            speaker = turn.get('speaker', 'unknown')
            text = turn.get('text', '')
            text_parts.append(f"{speaker}: {text}")
        full_text = " ".join(text_parts)
        
        tokens = tokenizer(full_text, truncation=True, max_length=512)
        token_len = len(tokens['input_ids'])
        lengths.append(token_len)
        
        if i == 0:
            print(f"\n{f} 샘플 1:")
            print(f"  턴 수: {len(dialog)}")
            print(f"  토큰 수: {token_len}")
            print(f"  Truncated: {'Yes (512)' if token_len == 512 else 'No'}")
    
    avg_len = sum(lengths) / len(lengths)
    print(f"{f} 평균 (3개 샘플): {avg_len:.0f} tokens")

# 전처리 데이터 확인
print("\n" + "=" * 60)
print("전처리 데이터 토큰 길이 확인 (요약+윈도우6)")
print("=" * 60)

for f in file_mapping.keys():
    data = json.load(open(f'data/processed/{f}_processed.json', encoding='utf-8'))
    
    # 첫 3개 샘플 확인
    lengths = []
    for i, session in enumerate(data[:3]):
        # 전처리 형식: summary + context_turns + response
        summary_text = "\n".join(session['summary'])
        context_text = " ".join([f"{t['speaker']}: {t['text']}" for t in session['context_turns']])
        response_text = session['response']
        
        full_text = f"[SUMMARY]\n{summary_text}\n\n[CONTEXT]\n{context_text}\n\n[RESPONSE]\n{response_text}"
        
        tokens = tokenizer(full_text, truncation=True, max_length=512)
        token_len = len(tokens['input_ids'])
        lengths.append(token_len)
        
        if i == 0:
            print(f"\n{f} 샘플 1:")
            print(f"  요약: {len(session['summary'])} bullets")
            print(f"  컨텍스트: {len(session['context_turns'])} turns")
            print(f"  토큰 수: {token_len}")
            print(f"  Truncated: {'Yes (512)' if token_len == 512 else 'No'}")
    
    avg_len = sum(lengths) / len(lengths)
    print(f"{f} 평균 (3개 샘플): {avg_len:.0f} tokens")
