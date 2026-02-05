"""
V3 vs V4 비교 중 위반 케이스만 추출
"""
import json

def main():
    # Load comparison
    with open('compare_v3_v4.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter violations (either V3 or V4 detected violation)
    violations = []
    
    for comp in data['comparisons']:
        v3_label = comp['v3_judgment']['label']
        v4_label = comp['v4_judgment']['label']
        
        # If either V3 or V4 detected violation
        if v3_label != 'normal' or v4_label != 'normal':
            violations.append(comp)
    
    # Count stats
    v3_violations = sum(1 for v in violations if v['v3_judgment']['label'] != 'normal')
    v4_violations = sum(1 for v in violations if v['v4_judgment']['label'] != 'normal')
    both_violations = sum(1 for v in violations if v['v3_judgment']['label'] != 'normal' and v['v4_judgment']['label'] != 'normal')
    
    # Create output
    output = {
        "summary": {
            "total_violations": len(violations),
            "v3_violations": v3_violations,
            "v4_violations": v4_violations,
            "both_detected": both_violations,
            "v3_only": v3_violations - both_violations,
            "v4_only": v4_violations - both_violations
        },
        "violations": violations
    }
    
    # Save
    with open('violations_v3_v4.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("위반 케이스 추출 완료")
    print("=" * 80)
    print(f"\n전체 위반 케이스: {len(violations)}개")
    print(f"  - V3가 감지: {v3_violations}개")
    print(f"  - V4가 감지: {v4_violations}개")
    print(f"  - 둘 다 감지: {both_violations}개")
    print(f"  - V3만 감지: {v3_violations - both_violations}개")
    print(f"  - V4만 감지: {v4_violations - both_violations}개")
    
    print(f"\n✅ 저장 완료: violations_v3_v4.json")
    
    # Print label distribution
    print(f"\n위반 레이블 분포:")
    v3_labels = {}
    v4_labels = {}
    for v in violations:
        v3_label = v['v3_judgment']['label']
        v4_label = v['v4_judgment']['label']
        if v3_label != 'normal':
            v3_labels[v3_label] = v3_labels.get(v3_label, 0) + 1
        if v4_label != 'normal':
            v4_labels[v4_label] = v4_labels.get(v4_label, 0) + 1
    
    print(f"\n  V3 위반:")
    for label, count in sorted(v3_labels.items()):
        print(f"    {label.upper()}: {count}개")
    
    print(f"\n  V4 위반:")
    for label, count in sorted(v4_labels.items()):
        print(f"    {label.upper()}: {count}개")

if __name__ == '__main__':
    main()
