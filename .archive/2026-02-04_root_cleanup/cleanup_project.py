"""
프로젝트 정리: 불필요한 파일 아카이브
"""
import os
import shutil
from pathlib import Path

# 아카이브 폴더 생성
archive_dir = ".archive/2026-02-04_cleanup"
os.makedirs(archive_dir, exist_ok=True)

print("="*80)
print("📁 프로젝트 정리 시작")
print("="*80)

# 1. 삭제할 프롬프트 파일 (V7 제외)
prompt_files_to_archive = [
    "src/llm/prompts.py",
    "src/llm/prompts_v2.py",
    "src/llm/prompts_v3.py",
    "src/llm/prompts_v4.py",
    "src/llm/prompts_v5.py",
    "src/llm/prompts_v6.py",
    "src/llm/prompts_v8.py",
    "src/llm/prompts_v9.py",
    "src/llm/prompts_v10.py",
]

# 2. 삭제할 테스트/분석 스크립트
script_files_to_archive = [
    "test_judge_v7.py",
    "test_judge_v8.py",
    "test_judge_v9.py",
    "check_first_speaker.py",
    "check_judge_diff.py",
    "check_v2_cases.py",
    "check_v9_session28.py",
    "compare_v7_v8_v9.py",
    "analyze_esconv_usage.py",
    "analyze_esconv_reuse.py",
    "fix_esconv_matching.py",
    "trace_generation_process.py",
    "test_v10_synthetic_sample.py",
    "analyze_v7_results.py",
    "examine_v2_cases.py",
    "extract_v4_v5_sessions.py",
    "view_v4_v5_candidates.py",
    "test_v7_single.py",
    "show_generated_jsons.py",
    "separate_v5_sessions.py",
    "detect_v4_v5_triggers.py",
]

# 3. 삭제할 JSON 파일
json_files_to_archive = [
    "test_judge_v7_100.json",
    "test_judge_v8_100.json",
    "test_judge_v9_100.json",
    "test_v10_synthetic_sample_results.json",
    "evaluate_prefixes_v7_sessions.json",
    "esconv_v4_candidates.json",
    "esconv_v5_candidates.json",
]

# 파일 이동
moved_count = 0
for file_list in [prompt_files_to_archive, script_files_to_archive, json_files_to_archive]:
    for file_path in file_list:
        if os.path.exists(file_path):
            dest = os.path.join(archive_dir, os.path.basename(file_path))
            shutil.move(file_path, dest)
            print(f"  Moved: {file_path} -> {archive_dir}/")
            moved_count += 1
        else:
            print(f"  Skip (not found): {file_path}")

print(f"\n✅ {moved_count}개 파일 이동 완료")

# 4. esconv_random_prefixes.json 이름 변경
old_name = "esconv_random_prefixes.json"
new_name = "ESConv_1300_prefixes.json"

if os.path.exists(old_name):
    os.rename(old_name, new_name)
    print(f"\n✅ 파일 이름 변경:")
    print(f"   {old_name} -> {new_name}")
    print(f"   (이제부터 이 파일이 실질적 원본 데이터입니다)")

print("\n" + "="*80)
print("📊 정리 완료 요약")
print("="*80)
print(f"아카이브 폴더: {archive_dir}/")
print(f"이동된 파일: {moved_count}개")
print("\n✅ 유지된 중요 파일:")
print("   - src/llm/prompts_v7.py (현재 사용 중인 judge 프롬프트)")
print("   - ESConv.json (ESConv 원본 1300개)")
print("   - ESConv_1300_prefixes.json (12-20턴 prefix, 실질적 원본)")
print("   - ESConv_v5_sessions.json (V5 트리거 있는 75개 세션)")
print("   - ESConv_normal_sessions.json (V5 트리거 없는 1225개 세션)")
print("   - esconv_v5_triggers.json (V5 트리거 탐지 결과)")
print("   - evaluate_prefixes_v7_results.json (V7 평가 결과)")
print("="*80)
