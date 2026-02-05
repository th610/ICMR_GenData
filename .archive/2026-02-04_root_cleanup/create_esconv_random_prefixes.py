import json
import random

# ESConv 로드
esconv = json.load(open('ESConv.json', encoding='utf-8'))

print('='*70)
print('ESConv PREFIX 데이터셋 생성 (랜덤 시작, supporter 끝)')
print('='*70)
print(f'원본 ESConv 세션: {len(esconv)}개')
print()

# PREFIX 추출 함수 (랜덤 시작)
def extract_random_prefix(dialog, min_turns=12, max_turns=20):
    """
    랜덤 시작 위치에서 12-20턴 prefix 추출 (마지막은 supporter)
    """
    if len(dialog) < min_turns:
        return None, None, None
    
    # 모든 가능한 (start, end) 조합 찾기
    valid_segments = []
    
    for start_idx in range(len(dialog)):
        for length in range(min_turns, max_turns + 1):
            end_idx = start_idx + length
            
            # 범위 체크
            if end_idx > len(dialog):
                break
            
            # 마지막 턴이 supporter인지 확인
            if dialog[end_idx - 1]['speaker'] == 'supporter':
                valid_segments.append((start_idx, end_idx, length))
    
    if not valid_segments:
        return None, None, None
    
    # 랜덤 선택
    start_idx, end_idx, length = random.choice(valid_segments)
    prefix_dialog = dialog[start_idx:end_idx]
    
    return prefix_dialog, start_idx, length


# 각 ESConv 세션에서 prefix 추출
random.seed(42)
prefix_dataset = []
skipped = 0

for idx, session in enumerate(esconv):
    prefix_dialog, start_idx, prefix_length = extract_random_prefix(session['dialog'])
    
    if prefix_dialog is None:
        skipped += 1
        continue
    
    # 새 샘플 생성
    prefix_sample = {
        'esconv_session_id': idx,
        'start_turn': start_idx,
        'end_turn': start_idx + prefix_length - 1,
        'prefix_length': prefix_length,
        'original_length': len(session['dialog']),
        'situation': session['situation'],
        'dialog': prefix_dialog,
        'source': 'esconv_random_prefix'
    }
    
    # 원본 메타데이터 복사
    if 'emotion_type' in session:
        prefix_sample['emotion_type'] = session['emotion_type']
    if 'problem_type' in session:
        prefix_sample['problem_type'] = session['problem_type']
    if 'experience_type' in session:
        prefix_sample['experience_type'] = session['experience_type']
    
    prefix_dataset.append(prefix_sample)
    
    if (idx + 1) % 200 == 0:
        print(f'처리 중: {idx + 1}/{len(esconv)}...')

print()
print(f'생성 완료: {len(prefix_dataset)}개')
print(f'스킵: {skipped}개 (12턴 미만 또는 supporter 끝 불가)')
print()

# 통계
prefix_lengths = [s['prefix_length'] for s in prefix_dataset]
start_positions = [s['start_turn'] for s in prefix_dataset]

print('PREFIX 길이 분포:')
for length in range(12, 21):
    count = sum(1 for l in prefix_lengths if l == length)
    print(f'  {length}턴: {count}개')

print()
print('시작 위치 통계:')
print(f'  0부터 시작: {sum(1 for s in start_positions if s == 0)}개')
print(f'  중간 시작: {sum(1 for s in start_positions if s > 0)}개')
print(f'  평균 시작 위치: {sum(start_positions)/len(start_positions):.1f}턴')

print()

# 샘플 확인
print('='*70)
print('샘플 확인 (처음 5개)')
print('='*70)

for i, sample in enumerate(prefix_dataset[:5]):
    print(f'\n[샘플 {i+1}]')
    print(f'  esconv_session_id: {sample["esconv_session_id"]}')
    print(f'  situation: {sample["situation"][:60]}...')
    print(f'  원본 길이: {sample["original_length"]}턴')
    print(f'  추출 범위: turn {sample["start_turn"]} ~ {sample["end_turn"]} ({sample["prefix_length"]}턴)')
    print(f'  첫 턴: {sample["dialog"][0]["speaker"]}')
    print(f'  마지막 턴: {sample["dialog"][-1]["speaker"]}')
    if sample['start_turn'] > 0:
        print(f'  ⭐ 중간부터 시작!')

# 저장
output_path = 'esconv_random_prefixes.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(prefix_dataset, f, indent=2, ensure_ascii=False)

print()
print('='*70)
print(f'저장 완료: {output_path}')
print(f'총 {len(prefix_dataset)}개 샘플')
print('='*70)
