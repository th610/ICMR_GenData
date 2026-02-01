import json
from pathlib import Path

files = [
    ('data/generated/normal_400.json', 400, 'normal'),
    ('data/generated/v1_240.json', 240, 'v1'),
    ('data/generated/v2_160.json', 160, 'v2'),
    ('data/generated/v3_200.json', 200, 'v3'),
    ('data/pilot/v4_full_150.json', 150, 'v4'),
    ('data/pilot/v5_full_150.json', 150, 'v5')
]

print('='*70)
print('Generated Dataset Summary')
print('='*70)

total = 0
for filepath, expected, label in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    actual = len(data)
    total += actual
    
    status = '✅' if actual == expected else '❌'
    print(f'{status} {label.upper():8s} {Path(filepath).name:25s} {actual:4d} (expected: {expected})')

print('='*70)
print(f'Total: {total} sessions (expected: 1300)')
print('='*70)
