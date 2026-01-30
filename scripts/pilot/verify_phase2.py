import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.prompts_v2 import *

# Phase 2 검증
print('✅ prompts_v2.py import 성공\n')

print('프롬프트 확인:')
print(f'  V1_SYSTEM: {len(V1_SYSTEM)} chars')
print(f'  V2_SYSTEM: {len(V2_SYSTEM)} chars')
print(f'  V3_SYSTEM: {len(V3_SYSTEM)} chars')
print(f'  V4_SYSTEM: {len(V4_SYSTEM)} chars')
print(f'  V5_SYSTEM: {len(V5_SYSTEM)} chars')
print(f'  SUMMARY_SYSTEM: {len(SUMMARY_SYSTEM)} chars')

print('\n템플릿 확인:')
print(f'  V1_USER_TEMPLATE: {len(V1_USER_TEMPLATE)} chars')
print(f'  V2_USER_TEMPLATE: {len(V2_USER_TEMPLATE)} chars')
print(f'  V3_USER_TEMPLATE: {len(V3_USER_TEMPLATE)} chars')
print(f'  V4_USER_TEMPLATE: {len(V4_USER_TEMPLATE)} chars')
print(f'  V5_USER_TEMPLATE: {len(V5_USER_TEMPLATE)} chars')
print(f'  SUMMARY_USER_TEMPLATE: {len(SUMMARY_USER_TEMPLATE)} chars')

print('\nHelper 함수 확인:')
funcs = ['format_dialog', 'build_v1_prompt', 'build_v2_prompt', 'build_v3_prompt', 
         'build_v4_prompt', 'build_v5_prompt', 'build_summary_prompt']
for func in funcs:
    if func in globals():
        print(f'  ✅ {func}')
    else:
        print(f'  ❌ {func} 없음')

print('\n✅ Phase 2 완료! 프롬프트 작성 성공')
