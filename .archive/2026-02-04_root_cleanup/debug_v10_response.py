import json
import random
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_v10 import JUDGE_SYSTEM, build_judge_prompt
from src.llm.state_memory import STATE_MEMORY_SYSTEM, build_state_memory_prompt

random.seed(42)

# V1 데이터 하나만
v1_data = json.load(open('.archive/2026-02-02_generated/generated/v1_240.json', encoding='utf-8'))
sample = v1_data[0]

situation = sample['situation']
dialog = sample['dialog']

print(f"Dialog 길이: {len(dialog)}턴")
print(f"마지막 턴: {dialog[-1]}")
print()

# LLM 클라이언트
llm_client = OpenAIClient()

# STATE MEMORY 생성
state_memory_dialog = dialog[:-1]
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

state_memory = response.choices[0].message.content.strip()

print("="*60)
print("STATE MEMORY:")
print("="*60)
print(state_memory)
print()

# WINDOW
window = dialog[-7:]

# Judge 프롬프트
judge_prompt = build_judge_prompt(window, state_memory)

print("="*60)
print("JUDGE PROMPT (일부):")
print("="*60)
print(judge_prompt[:500])
print("...")
print()

# Judge 응답
response = llm_client.client.chat.completions.create(
    model=llm_client.model,
    messages=[
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt}
    ],
    temperature=0.0,
    max_tokens=300
)

judge_response = response.choices[0].message.content.strip()

print("="*60)
print("JUDGE RESPONSE:")
print("="*60)
print(judge_response)
print()

# 라벨 파싱
lines = judge_response.split('\n')
print(f"첫 줄: {lines[0]}")
label = lines[0].replace('Label:', '').strip()
print(f"파싱된 label: '{label}'")
