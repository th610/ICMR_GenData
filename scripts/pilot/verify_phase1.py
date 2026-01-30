import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_yaml

# Phase 1 검증
config = load_yaml('configs/pilot.yaml')

print('✅ YAML 파싱 성공')
print(f'\n주요 설정:')
print(f'  Total sessions: {config["pilot"]["total_sessions"]}')
print(f'  Per class: {config["pilot"]["sessions_per_class"]}')
print(f'  LLM model: {config["llm"]["model"]}')
print(f'  Split: train={config["split"]["train"]}, val={config["split"]["val"]}, test={config["split"]["test"]}')
print(f'  Labels: {config["labels"]}')
print(f'  Normal sessions: {config["normal"]["num_sessions"]}')
print(f'  V1~V3 per class: {config["v1_v3"]["num_per_class"]}')
print(f'  V4/V5 per class: {config["v4_v5"]["num_per_class"]}')

print('\n✅ Phase 1 완료! 설정 파일 작성 성공')
