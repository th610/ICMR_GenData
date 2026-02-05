"""
V3 vs V4 결과 비교
"""
import json

def main():
    # Load results
    with open('test_judge_v3_100.json', 'r', encoding='utf-8') as f:
        v3_results = json.load(f)
    
    with open('test_judge_v4_100.json', 'r', encoding='utf-8') as f:
        v4_results = json.load(f)
    
    # Create comparison - ALL cases
    all_comparisons = []
    agreements = 0
    disagreements = 0
    
    for v3, v4 in zip(v3_results, v4_results):
        session_id = v3['session_id']
        
        # Check if session IDs match
        assert session_id == v4['session_id'], f"Session ID mismatch: {session_id} vs {v4['session_id']}"
        
        v3_label = v3['label']
        v4_label = v4['label']
        
        # Check agreement
        is_agreement = (v3_label == v4_label)
        
        if is_agreement:
            agreements += 1
        else:
            disagreements += 1
            
        # Save ALL cases (not just disagreements)
        all_comparisons.append({
            "session_id": session_id,
            "window_length": v3['window_length'],
            "dialog": v3['dialog'],
            "v3_judgment": {
                "label": v3['label'],
                "reason": v3['reason'],
                "evidence": v3.get('evidence', 'N/A'),
                "confidence": v3['confidence']
            },
            "v4_judgment": {
                "label": v4['label'],
                "reason": v4['reason'],
                "evidence": v4.get('evidence', 'N/A'),
                "confidence": v4['confidence']
            },
            "agreement": is_agreement
        })
    
    # Print summary
    print("=" * 80)
    print("V3 vs V4 비교 결과")
    print("=" * 80)
    print(f"\n전체: {len(v3_results)}개")
    print(f"✅ 일치: {agreements}개 ({agreements/len(v3_results)*100:.1f}%)")
    print(f"❌ 불일치: {disagreements}개 ({disagreements/len(v3_results)*100:.1f}%)")
    
    # Save to file
    output = {
        "summary": {
            "total": len(v3_results),
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": f"{agreements/len(v3_results)*100:.1f}%"
        },
        "comparisons": all_comparisons
    }
    
    with open('compare_v3_v4.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 저장 완료: compare_v3_v4.json ({len(all_comparisons)}개 케이스)")
    
    # Print disagreement breakdown
    if disagreements > 0:
        print(f"\n불일치 케이스 분석:")
        disagreement_types = {}
        for comp in all_comparisons:
            if not comp['agreement']:
                v3_label = comp['v3_judgment']['label']
                v4_label = comp['v4_judgment']['label']
                dtype = f"{v3_label}_vs_{v4_label}"
                disagreement_types[dtype] = disagreement_types.get(dtype, 0) + 1
        
        for dtype, count in sorted(disagreement_types.items()):
            print(f"  {dtype}: {count}개")

if __name__ == '__main__':
    main()
