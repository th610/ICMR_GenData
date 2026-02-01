import json

# 전처리 데이터 확인
files = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
print("=" * 60)
print("전처리 데이터")
print("=" * 60)
total = 0
for f in files:
    data = json.load(open(f'data/processed/{f}_processed.json', encoding='utf-8'))
    total += len(data)
    print(f'{f:10s}: {len(data):4d} sessions')
print(f'\nTotal: {total} sessions')

# Judge 결과 확인
print("\n" + "=" * 60)
print("Judge 평가 결과")
print("=" * 60)
judge_data = json.load(open('data/final/judge_all_results.json', encoding='utf-8'))
summary = judge_data['summary']
print(f"Overall: {summary['overall_correct']}/{summary['overall_total']} ({summary['overall_accuracy']:.1f}%)\n")
for cls, stats in summary['by_class'].items():
    print(f"{cls:10s}: {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']:5.1f}%)")
