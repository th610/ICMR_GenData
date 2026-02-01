"""
Organize ESConv Judge results by label into separate folders
"""
from pathlib import Path
from src.utils import load_json, save_json


def main():
    print("=" * 80)
    print("ESConv Judge 결과 폴더별 정리")
    print("=" * 80)
    
    # Load judge results
    judge_results_path = "data/pilot/judge_esconv_full_1300.json"
    print(f"\n📂 {judge_results_path} 로드 중...")
    sessions = load_json(judge_results_path)
    print(f"   총 세션: {len(sessions)}개")
    
    # Create output directories
    base_dir = Path("data/esconv_judged")
    normal_dir = base_dir / "normal"
    violations_dir = base_dir / "violations"
    
    dirs = {
        "Normal": normal_dir,
        "V1": violations_dir / "v1",
        "V2": violations_dir / "v2",
        "V3": violations_dir / "v3",
        "V4": violations_dir / "v4",
        "V5": violations_dir / "v5"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Organize by label
    label_counts = {}
    
    for label, dir_path in dirs.items():
        filtered = [s for s in sessions if s.get('judge_label') == label]
        label_counts[label] = len(filtered)
        
        if filtered:
            output_path = dir_path / f"{label.lower()}_sessions.json"
            save_json(filtered, str(output_path))
            print(f"   ✅ {label:8s} {len(filtered):4d}개 → {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("정리 완료")
    print("=" * 80)
    print(f"\n폴더 구조:")
    print(f"  data/esconv_judged/")
    print(f"    ├── normal/           ({label_counts['Normal']}개)")
    print(f"    └── violations/")
    print(f"        ├── v1/           ({label_counts['V1']}개)")
    print(f"        ├── v2/           ({label_counts['V2']}개)")
    print(f"        ├── v3/           ({label_counts['V3']}개)")
    print(f"        ├── v4/           ({label_counts['V4']}개)")
    print(f"        └── v5/           ({label_counts['V5']}개)")
    
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
