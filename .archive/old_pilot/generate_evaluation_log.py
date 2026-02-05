"""
Judge 평가 결과를 분석하고 로그로 저장

judge_results_summary_window.json을 읽어서:
1. 통계 요약
2. 클래스별 정확도
3. 오판 패턴 분석
4. 실패 사례 상세
→ EVALUATION_LOG.md로 저장
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter


def analyze_judge_results(results_file: Path):
    """Judge 결과 분석"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    timestamp = data.get('timestamp', 'Unknown')
    results = data.get('results', {})
    
    # 통계 수집
    stats = {
        'total': 0,
        'match': 0,
        'mismatch': 0,
        'by_class': {},
        'confusion_matrix': Counter(),
        'mismatches': []
    }
    
    for class_name, sessions in results.items():
        class_total = len(sessions)
        class_match = sum(1 for s in sessions if s.get('matches'))
        class_mismatch = class_total - class_match
        
        stats['total'] += class_total
        stats['match'] += class_match
        stats['mismatch'] += class_mismatch
        
        stats['by_class'][class_name] = {
            'total': class_total,
            'match': class_match,
            'mismatch': class_mismatch,
            'accuracy': (class_match / class_total * 100) if class_total > 0 else 0
        }
        
        # Confusion matrix
        for session in sessions:
            expected = session.get('expected_label', 'Unknown')
            predicted = session.get('predicted_label', 'Unknown')
            if not session.get('matches'):
                stats['confusion_matrix'][(expected, predicted)] += 1
                stats['mismatches'].append({
                    'session_id': session.get('session_id'),
                    'expected': expected,
                    'predicted': predicted,
                    'reason': session.get('reason', ''),
                    'confidence': session.get('confidence', '')
                })
    
    return stats, timestamp


def generate_log_markdown(stats: dict, timestamp: str, output_file: Path):
    """Markdown 로그 생성"""
    
    lines = []
    lines.append("# Judge 평가 결과 로그")
    lines.append(f"\n**평가 일시**: {timestamp}")
    lines.append(f"\n---\n")
    
    # 전체 통계
    lines.append("## 📊 전체 통계\n")
    accuracy = (stats['match'] / stats['total'] * 100) if stats['total'] > 0 else 0
    lines.append(f"- **전체 세션**: {stats['total']}개")
    lines.append(f"- **정확히 판정**: {stats['match']}개")
    lines.append(f"- **오판**: {stats['mismatch']}개")
    lines.append(f"- **전체 정확도**: {accuracy:.1f}%\n")
    
    # 클래스별 정확도
    lines.append("## 📈 클래스별 정확도\n")
    lines.append("| 클래스 | 전체 | 정확 | 오판 | 정확도 |")
    lines.append("|--------|------|------|------|--------|")
    
    for class_name in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
        if class_name in stats['by_class']:
            cls_stats = stats['by_class'][class_name]
            lines.append(
                f"| {class_name.upper():6s} | "
                f"{cls_stats['total']:4d} | "
                f"{cls_stats['match']:4d} | "
                f"{cls_stats['mismatch']:4d} | "
                f"{cls_stats['accuracy']:6.1f}% |"
            )
    
    lines.append("")
    
    # Confusion Matrix
    lines.append("## 🔄 오판 패턴 (Confusion Matrix)\n")
    if stats['confusion_matrix']:
        lines.append("| 실제 라벨 | Judge 판정 | 횟수 |")
        lines.append("|-----------|------------|------|")
        for (expected, predicted), count in stats['confusion_matrix'].most_common():
            lines.append(f"| {expected:9s} | {predicted:10s} | {count:4d} |")
        lines.append("")
    else:
        lines.append("*(오판 없음)*\n")
    
    # 실패 사례 상세
    lines.append("## ❌ 실패 사례 상세\n")
    
    if stats['mismatches']:
        # 클래스별로 그룹화
        by_class = {}
        for m in stats['mismatches']:
            expected = m['expected']
            if expected not in by_class:
                by_class[expected] = []
            by_class[expected].append(m)
        
        for class_name in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
            if class_name in by_class:
                lines.append(f"### {class_name}\n")
                for m in by_class[class_name]:
                    lines.append(f"**[{m['session_id']}]** {m['expected']} → {m['predicted']} ({m['confidence']})")
                    lines.append(f"- 이유: {m['reason']}\n")
    else:
        lines.append("*(실패 사례 없음)*\n")
    
    # 결론 및 제안
    lines.append("---\n")
    lines.append("## 💡 분석 및 제안\n")
    
    # 성공한 클래스
    success_classes = [
        cls for cls, data in stats['by_class'].items()
        if data['accuracy'] >= 80
    ]
    
    # 실패한 클래스
    fail_classes = [
        cls for cls, data in stats['by_class'].items()
        if data['accuracy'] < 50
    ]
    
    if success_classes:
        lines.append(f"**✅ 성공 클래스** (정확도 ≥80%): {', '.join([c.upper() for c in success_classes])}\n")
    
    if fail_classes:
        lines.append(f"**❌ 개선 필요** (정확도 <50%): {', '.join([c.upper() for c in fail_classes])}\n")
    
    lines.append("\n**다음 단계:**")
    
    if accuracy >= 70:
        lines.append("- ✅ 전체 정확도 양호 - 전체 데이터 생성 진행 가능")
    elif accuracy >= 50:
        lines.append("- ⚠️ 실패 클래스 생성 프롬프트 개선 필요")
        lines.append("- 개선 후 재생성 및 재평가 권장")
    else:
        lines.append("- ❌ 생성 프롬프트 전면 재검토 필요")
        lines.append("- Judge 프롬프트와 생성 프롬프트 정합성 확인")
    
    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return '\n'.join(lines)


def main():
    """메인 함수"""
    
    results_file = Path("data/pilot/judge_results_summary_window.json")
    output_file = Path("EVALUATION_LOG.md")
    
    if not results_file.exists():
        print(f"❌ Judge 결과 파일 없음: {results_file}")
        return
    
    print("=" * 80)
    print("Judge 평가 결과 분석 및 로그 생성")
    print("=" * 80)
    print()
    
    # 분석
    print("📊 결과 분석 중...")
    stats, timestamp = analyze_judge_results(results_file)
    
    # 로그 생성
    print("📝 로그 생성 중...")
    log_content = generate_log_markdown(stats, timestamp, output_file)
    
    print(f"\n✅ 로그 저장 완료: {output_file}")
    print()
    
    # 간단한 요약 출력
    print("=" * 80)
    print("📊 요약")
    print("=" * 80)
    accuracy = (stats['match'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"전체 정확도: {accuracy:.1f}% ({stats['match']}/{stats['total']})")
    print()
    
    print("클래스별:")
    for class_name in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
        if class_name in stats['by_class']:
            cls_stats = stats['by_class'][class_name]
            status = "✅" if cls_stats['accuracy'] >= 80 else "⚠️" if cls_stats['accuracy'] >= 50 else "❌"
            print(f"  {status} {class_name.upper():6s}: {cls_stats['accuracy']:5.1f}% ({cls_stats['match']}/{cls_stats['total']})")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
