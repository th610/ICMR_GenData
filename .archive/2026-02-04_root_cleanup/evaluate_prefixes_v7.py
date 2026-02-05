"""
ESConv Random Prefix 데이터셋을 V7 Judge로 평가
모든 샘플은 Normal로 분류되어야 함 (ESConv original data)
"""
import json
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_v7 import JUDGE_SYSTEM, build_judge_prompt

def judge_with_v7(dialog, client):
    """V7 judge로 평가 (dialog는 list of dict)"""
    # dialog는 이미 {'speaker': ..., 'content': ...} 형식
    # speaker를 supporter/seeker로 변환
    converted_dialog = []
    for turn in dialog:
        speaker = 'supporter' if turn['speaker'] == 'supporter' else 'seeker'
        converted_dialog.append({
            'speaker': speaker,
            'content': turn['content']
        })
    
    user_prompt = build_judge_prompt(converted_dialog)
    
    result = client.call(
        system_prompt=JUDGE_SYSTEM,
        user_prompt=user_prompt
    )
    
    label = result.get('label', 'Unknown')
    
    # Normal이면 label만 반환, 위반이면 전체 정보 반환
    if label == 'Normal' or label == 'normal':
        return {'label': label.capitalize()}
    else:
        return {
            'label': label.capitalize(),
            'reason': result.get('reason', ''),
            'evidence': result.get('evidence', ''),
            'full_response': str(result)
        }

def main():
    # 데이터 로드
    print("Loading esconv_random_prefixes.json...")
    with open("esconv_random_prefixes.json", "r", encoding="utf-8") as f:
        prefixes = json.load(f)
    
    print(f"Total samples: {len(prefixes)}")
    print("Starting evaluation...")
    
    # OpenAI 클라이언트 초기화
    client = OpenAIClient(model="gpt-4o-mini", temperature=0.0)
    
    # 평가 시작
    results = []
    violations = []  # 위반 케이스만 별도 저장
    stats = {"Normal": 0, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0}
    
    for idx, sample in enumerate(prefixes):
        session_id = sample['esconv_session_id']
        prefix_length = sample['prefix_length']
        
        # V7 judge 호출 (dialog를 직접 전달)
        try:
            judge_result = judge_with_v7(sample['dialog'], client)
            
            # judge_result가 dict인지 확인
            if isinstance(judge_result, dict):
                judgment = judge_result.get('label', 'Unknown')
            else:
                # 예상치 못한 형식
                print(f"Unexpected result type: {type(judge_result)}, value: {judge_result}")
                raise ValueError(f"Invalid judge result format")
            
            stats[judgment] = stats.get(judgment, 0) + 1
            
            # 기본 결과 (인덱스만)
            result = {
                "sample_idx": idx,
                "esconv_session_id": session_id,
                "prefix_length": prefix_length,
                "judgment": judgment,
                "correct": judgment == "Normal"
            }
            
            # 위반인 경우만 상세 정보 추가
            if judgment != "Normal":
                result.update({
                    "reason": judge_result.get('reason', ''),
                    "evidence": judge_result.get('evidence', ''),
                    "full_response": judge_result.get('full_response', '')
                })
                violations.append(result)
            
            results.append(result)
            
            # 진행상황 출력 (100개마다)
            if (idx + 1) % 100 == 0:
                accuracy = sum(1 for r in results if r['correct']) / len(results) * 100
                print(f"Progress: {idx + 1}/{len(prefixes)} | Accuracy: {accuracy:.1f}% | Normal: {stats['Normal']}")
                
        except Exception as e:
            print(f"Error on sample {idx} (session {session_id}): {e}")
            result = {
                "sample_idx": idx,
                "esconv_session_id": session_id,
                "prefix_length": prefix_length,
                "error": str(e)
            }
            results.append(result)
    
    # 최종 결과
    correct_count = sum(1 for r in results if r.get('correct', False))
    total_evaluated = sum(1 for r in results if 'judgment' in r)
    accuracy = correct_count / total_evaluated * 100 if total_evaluated > 0 else 0
    
    print("\n" + "="*60)
    print("V7 Evaluation Results on ESConv Random Prefixes")
    print("="*60)
    print(f"Total samples: {len(prefixes)}")
    print(f"Evaluated: {total_evaluated}")
    print(f"Errors: {len(results) - total_evaluated}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_evaluated})")
    print("\nJudgment Distribution:")
    for label, count in sorted(stats.items()):
        if count > 0:
            pct = count / total_evaluated * 100 if total_evaluated > 0 else 0
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    # 결과 저장
    output = {
        "total_samples": len(prefixes),
        "evaluated": total_evaluated,
        "errors": len(results) - total_evaluated,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "stats": stats,
        "results": results
    }
    
    with open("evaluate_prefixes_v7_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: evaluate_prefixes_v7_results.json")
    
    # 세션별 위반 분석
    from collections import defaultdict
    session_violations = defaultdict(list)
    for v in violations:
        session_violations[v['esconv_session_id']].append({
            'sample_idx': v['sample_idx'],
            'prefix_length': v['prefix_length'],
            'judgment': v['judgment']
        })
    
    print(f"\n📊 Session-level Violation Analysis")
    print(f"Total sessions with violations: {len(session_violations)}")
    print(f"Total violation samples: {len(violations)}")
    
    if session_violations:
        # 위반이 많은 세션 Top 10
        sorted_sessions = sorted(session_violations.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\nTop 10 sessions with most violations:")
        for session_id, viols in sorted_sessions[:10]:
            violation_types = [v['judgment'] for v in viols]
            print(f"  Session {session_id}: {len(viols)} violations - {violation_types}")
        
        # 세션별 위반 정보 저장
        session_summary = {
            str(session_id): {
                'violation_count': len(viols),
                'violations': viols
            }
            for session_id, viols in session_violations.items()
        }
        
        with open("evaluate_prefixes_v7_sessions.json", "w", encoding="utf-8") as f:
            json.dump(session_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nSession analysis saved to: evaluate_prefixes_v7_sessions.json")
    
    # 오분류 샘플 분석
    misclassified = [r for r in results if r.get('correct') == False]
    if misclassified:
        print(f"\n⚠️ Misclassified Samples: {len(misclassified)}")
        print("First 10 misclassifications:")
        for r in misclassified[:10]:
            print(f"  Sample {r['sample_idx']} | Session {r['esconv_session_id']} (len={r['prefix_length']}): {r['judgment']}")

if __name__ == "__main__":
    main()
