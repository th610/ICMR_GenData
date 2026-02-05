import json
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_v10 import JUDGE_SYSTEM, build_judge_prompt
from src.llm.state_memory import STATE_MEMORY_SYSTEM, build_state_memory_prompt

# V3 샘플 로드
v3_data = json.load(open('.archive/2026-02-02_generated/generated/v3_200.json', encoding='utf-8'))
sample = v3_data[0]

situation = sample['situation']
dialog = sample['dialog']
violation_turn = sample['violation_turn_index']

print('='*60)
print('V3 샘플 디버깅')
print('='*60)
print(f'Situation: {situation}')
print(f'Dialog 길이: {len(dialog)}턴')
print(f'위반 턴: {violation_turn}')
print()

# STATE MEMORY 생성 (마지막 턴 제외)
state_memory_dialog = dialog[:-1]
print(f'STATE MEMORY 생성 범위: turn 0 ~ {len(state_memory_dialog)-1}')
print()

llm_client = OpenAIClient()

# STATE MEMORY 생성
prompt = build_state_memory_prompt(situation, state_memory_dialog)
response = llm_client.client.chat.completions.create(
    model=llm_client.model,
    messages=[
        {"role": "system", "content": STATE_MEMORY_SYSTEM},
        {"role": "user", "content": prompt}
    ],
    temperature=0.0,
    max_tokens=300
)

content = response.choices[0].message.content.strip()
result = json.loads(content)
state_memory = result.get('state_memory', '')

print('='*60)
print('STATE MEMORY:')
print('='*60)
print(state_memory)
print()

# WINDOW (마지막 7턴)
window = dialog[-7:]
print('='*60)
print('WINDOW (마지막 7턴):')
print('='*60)
for i, turn in enumerate(window[:-1]):  # TARGET 제외
    print(f'{turn["speaker"]}: {turn["content"][:100]}...')
print()

# TARGET
target = dialog[-1]
print('='*60)
print('TARGET (위반 turnindex:', violation_turn, '):')
print('='*60)
print(f'{target["speaker"]}: {target["content"]}')
print()

# Judge 평가
judge_prompt = build_judge_prompt(window, state_memory)

response = llm_client.client.chat.completions.create(
    model=llm_client.model,
    messages=[
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt}
    ],
    temperature=0.0,
    max_tokens=300
)

judge_content = response.choices[0].message.content.strip()
judge_result = json.loads(judge_content)

print('='*60)
print('JUDGE 결과:')
print('='*60)
print(f'Label: {judge_result["label"]}')
print(f'Reason: {judge_result["reason"]}')
print(f'Evidence: {judge_result["evidence"]}')
print()

print(f'Ground Truth: V3')
print(f'Prediction: {judge_result["label"]}')
print(f'{"✅ 정답" if judge_result["label"] == "V3" else "❌ 오답"}')
