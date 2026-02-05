# Dataset Generation 작업 정리

**작성일**: 2026-02-04  
**프로젝트**: ICMR_GenData - Empathy Violation Detection Dataset

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [작업 히스토리](#작업-히스토리)
3. [현재 상태](#현재-상태)
4. [다음 단계](#다음-단계)

---

## 🎯 프로젝트 개요

### 목표
ESConv 데이터셋을 기반으로 **공감 위반 탐지를 위한 1300개의 합성 대화 샘플 생성**

### 데이터 분포
- **Normal**: 525개 (40.4%) - 위반 없음
- **V1 (Context)**: 200개 (15.4%) - 맥락 무시
- **V2 (Autonomy)**: 200개 (15.4%) - 자율성 침해  
- **V3 (Empathy-only)**: 200개 (15.4%) - 공감만 있고 조언 없음
- **V4 (Reality)**: 100개 (7.7%) - 현실 왜곡
- **V5 (Crisis)**: 75개 (5.8%) - 위기 상황 실패

**총**: 1300개 샘플, **1300개 유니크 세션** (중복 없음)

### 데이터 소스
- `ESConv_normal_prefixes.json`: 1225개 일반 세션
- `ESConv_v5_prefixes.json`: 75개 위기 세션 (crisis trigger 포함)

---

## 📜 작업 히스토리

### Phase 1: 프롬프트 개발 (완료 ✓)

**작업 내용**:
- V1-V5, Normal 프롬프트 anti-templating 강화
- 각 위반 타입별 시스템 프롬프트 작성
- Build function 구현 (`prompts.py`)

**주요 개선점**:
- 템플릿화 방지 지침 추가
- Prefix와의 자연스러운 연결 강조
- 위반의 명확성과 미묘함 균형

---

### Phase 2: Pilot Test (완료 ✓)

**스크립트**: `pilot_test_generation.py`  
**결과**: 30/30 성공 (100%)  
- 각 라벨당 5개씩 샘플링 (seed=42)
- 모든 프롬프트 정상 작동 확인

**출력**: `pilot_test_samples.json` (삭제됨)

---

### Phase 3: 첫 번째 전체 생성 시도 (실패 ❌)

**스크립트**: `generate_parallel.py`  
**전략**: 1300개를 5개 범위로 분할 (0-260, 260-520, ...)

**문제**:
- API Rate Limit 초과 (RPM: 500)
- 6개 병렬 프로세스가 과부하 유발
- 641/1300만 생성 후 중단

**결과**: 641개 불완전 샘플 생성 → 폐기

---

### Phase 4: Label-based 생성 (부분 성공 ⚠️)

**스크립트**: `generate_by_label.py`  
**전략**: 라벨별 독립 생성

**실행 결과**:
- ✅ V1: 200/200
- ✅ V2: 200/200
- ✅ V3: 200/200
- ✅ V4: 100/100
- ⚠️ V5: 74/75 (세션 947 누락)
- ⚠️ Normal: 145/525 (터미널 중단)

**문제 발견**:
1. **터미널 관리**: Agent가 실행 중인 터미널에서 명령 실행하여 프로세스 중단
2. **Normal 중단**: 145개 생성 후 KeyboardInterrupt

---

### Phase 5: Normal 복구 및 완료 (완료 ✓)

**전략**: Normal을 3개로 분할하여 병렬 생성

**스크립트**:
- `run_normal_part1.py`: 0-175 (175개)
- `run_normal_part2.py`: 175-350 (175개)
- `run_normal_part3.py`: 350-525 (175개)

**결과**:
- Part 1: 175/175 ✓
- Part 2: 175/175 ✓
- Part 3: 175/175 ✓
- **병합**: 525/525 ✓

**V5 수정**:
- `fix_v5.py`로 세션 947 생성
- V5: 75/75 ✓

**생성 완료**: 1299/1300 (99.9%)

---

### Phase 6: 중복 세션 문제 발견 (치명적 ❌)

**문제 분석**:
```python
# 각 라벨이 같은 normal_prefixes에서 다른 seed로 샘플링
seed_map = {"V1": 1, "V2": 2, "V3": 3, "V4": 4, "Normal": 5}
random.seed(seed_map[label])
prefixes = random.sample(all_prefixes, count)
```

**결과**:
- 생성된 샘플: 1299개
- **실제 유니크 세션: 918개만** ❌
- 중복 예시:
  - V1 & V2 overlap: 30 sessions
  - V1 & V3 overlap: 33 sessions
  - V1 & Normal overlap: 89 sessions
  - (총 11개 조합에서 중복 발생)

**원인**: 각 라벨이 독립적으로 샘플링했지만, 같은 pool에서 뽑아서 중복 발생

**요구사항 재확인**:
- "모든 다른 세션에서 위반 만들기"
- **1300개 완전히 유니크한 세션 사용** 필요

---

### Phase 7: 정리 및 재설계 (진행 중 🔄)

#### 7.1 파일 정리 (완료 ✓)

**삭제된 파일**:
- 생성 데이터: `generated_V1~V5.json`, `generated_Normal*.json`, `generated_part*.json`
- 임시 스크립트: `run_normal_part*.py`, `generate_normal_part*.py`, `continue_normal.py`, `fix_v5.py`, `check_sessions.py`, `merge_normal.py`, `generate_parallel.py`

**유지된 파일**:
- `generate_by_label.py`: 참고용
- `generate_full_1300.py`: 참고용
- `create_assignments.py`: 세션 할당 스크립트

#### 7.2 세션 재할당 (완료 ✓)

**스크립트**: `create_assignments.py`

**전략**:
```python
# 1. normal_prefixes 1225개를 섞음 (seed=42)
random.seed(42)
sampled_1225 = random.sample(normal_prefixes, 1225)

# 2. 중복 없이 분할
splits = {
    "Normal": sampled_1225[0:525],      # 0-525
    "V1": sampled_1225[525:725],        # 525-725
    "V2": sampled_1225[725:925],        # 725-925
    "V3": sampled_1225[925:1125],       # 925-1125
    "V4": sampled_1225[1125:1225],      # 1125-1225
    "V5": v5_prefixes                   # 75개 (별도)
}
```

**검증 결과**:
- ✅ 총 1300개 샘플
- ✅ 1300개 유니크 세션
- ✅ 라벨 간 중복 없음

**출력 파일**:
- `ESConv_v1_assigned.json` (200 sessions)
- `ESConv_v2_assigned.json` (200 sessions)
- `ESConv_v3_assigned.json` (200 sessions)
- `ESConv_v4_assigned.json` (100 sessions)
- `ESConv_normal_assigned.json` (525 sessions)
- `ESConv_v5_assigned.json` (75 sessions)
- `session_assignments.json` (할당 기록)

#### 7.3 새 생성 스크립트 작성 (완료 ✓)

**스크립트 목록**:

1. **gen_v1.py**: V1 200개 생성
   - 입력: `ESConv_v1_assigned.json`
   - 출력: `generated_V1.json`

2. **gen_v2.py**: V2 200개 생성
   - 입력: `ESConv_v2_assigned.json`
   - 출력: `generated_V2.json`

3. **gen_v3.py**: V3 200개 생성
   - 입력: `ESConv_v3_assigned.json`
   - 출력: `generated_V3.json`

4. **gen_v4.py**: V4 100개 생성
   - 입력: `ESConv_v4_assigned.json`
   - 출력: `generated_V4.json`

5. **gen_normal_part1.py**: Normal 175개 (0-175)
   - 입력: `ESConv_normal_assigned.json[0:175]`
   - 출력: `generated_Normal_part1.json`

6. **gen_normal_part2.py**: Normal 175개 (175-350)
   - 입력: `ESConv_normal_assigned.json[175:350]`
   - 출력: `generated_Normal_part2.json`

7. **gen_normal_part3.py**: Normal 175개 (350-525)
   - 입력: `ESConv_normal_assigned.json[350:525]`
   - 출력: `generated_Normal_part3.json`

**공통 특징**:
- OpenAI gpt-4o-mini 사용
- Temperature: 0.9
- Max tokens: 800
- Timeout: 60초
- 20개마다 progress 출력

---

## 📊 현재 상태

### ✅ 완료된 작업

1. **프롬프트 개발**: V1-V5, Normal 프롬프트 완성
2. **세션 할당**: 1300개 유니크 세션 중복 없이 분배
3. **Assigned 파일 생성**: 각 라벨별 prefix 파일 준비
4. **생성 스크립트 준비**: 7개 스크립트 작성 완료

### ⏳ 대기 중

**Phase 8: 최종 생성 실행**

#### 병렬 실행 계획 (3개씩)

**1차 배치** (약 200개씩):
```bash
python gen_v1.py       # 200개
python gen_v2.py       # 200개
python gen_v3.py       # 200개
```

**2차 배치**:
```bash
python gen_v4.py              # 100개
python gen_normal_part1.py    # 175개
python gen_normal_part2.py    # 175개
```

**3차 배치**:
```bash
python gen_normal_part3.py    # 175개
```

#### 예상 소요 시간

- V1, V2, V3: 각 ~20-30분 (병렬 실행)
- V4: ~10-15분
- Normal parts: 각 ~15-20분

**총 예상 시간**: 약 1.5-2시간

---

## 🎯 다음 단계

### 1. 최종 생성 실행

**명령어**:
```bash
# 1차 실행 (새 터미널 3개)
python gen_v1.py
python gen_v2.py  
python gen_v3.py

# 1차 완료 후 2차 실행
python gen_v4.py
python gen_normal_part1.py
python gen_normal_part2.py

# 2차 완료 후 3차 실행
python gen_normal_part3.py
```

### 2. Normal 파일 병합

**스크립트 작성 필요**:
```python
# merge_final_normal.py
import json

with open("generated_Normal_part1.json") as f:
    part1 = json.load(f)
with open("generated_Normal_part2.json") as f:
    part2 = json.load(f)
with open("generated_Normal_part3.json") as f:
    part3 = json.load(f)

merged = {
    "metadata": {
        "label": "Normal",
        "target_count": 525,
        "actual_count": len(part1["samples"]) + len(part2["samples"]) + len(part3["samples"])
    },
    "samples": part1["samples"] + part2["samples"] + part3["samples"]
}

with open("generated_Normal.json", 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
```

### 3. 최종 검증

**검증 스크립트**:
```python
# verify_final_dataset.py
import json

labels = ["V1", "V2", "V3", "V4", "V5", "Normal"]
all_sessions = set()

for label in labels:
    with open(f"generated_{label}.json") as f:
        data = json.load(f)
        sessions = {s["esconv_session_id"] for s in data["samples"]}
        all_sessions.update(sessions)
        print(f"{label}: {len(data['samples'])} samples, {len(sessions)} unique sessions")

print(f"\nTotal unique sessions: {len(all_sessions)}")
print(f"Target: 1300")
print(f"Status: {'✓ Complete' if len(all_sessions) == 1300 else '✗ Incomplete'}")

# Check for duplicates
total_samples = sum(len(json.load(open(f"generated_{l}.json"))["samples"]) for l in labels)
if total_samples == len(all_sessions):
    print("✓ No duplicate sessions across labels")
else:
    print(f"⚠️ Duplicates found: {total_samples - len(all_sessions)} sessions")
```

### 4. Train-Silver / Test-Gold 분할

**분할 계획**:
- Train-Silver: 1000 samples
- Test-Gold: 300 samples

**분포 유지**:
```python
splits = {
    "Normal": {"train": 445, "test": 80},
    "V1": {"train": 150, "test": 50},
    "V2": {"train": 140, "test": 60},
    "V3": {"train": 150, "test": 50},
    "V4": {"train": 70, "test": 30},
    "V5": {"train": 45, "test": 30}
}
```

### 5. 품질 검증

- 각 라벨당 10-20개 샘플 체크
- 템플릿화 여부 확인
- Prefix 연속성 확인
- 위반 명확성 확인

---

## 🔧 기술 스펙

### API 설정
- **Model**: gpt-4o-mini
- **Temperature**: 0.9
- **Max tokens**: 800
- **Timeout**: 60초
- **Retries**: 1회
- **Rate limit**: 500 RPM

### 병렬 실행 전략
- 최대 동시 실행: 3개 프로세스
- 이유: API rate limit 방지
- 각 프로세스는 독립된 터미널에서 background 실행

---

## 📝 교훈 및 개선사항

### 문제점 및 해결

1. **API Rate Limiting**
   - 문제: 6개 병렬 → 500 RPM 초과
   - 해결: 3개씩 순차 배치

2. **터미널 관리**
   - 문제: 실행 중인 터미널에서 명령 실행 → 프로세스 중단
   - 해결: 항상 새 터미널 사용, isBackground=true

3. **세션 중복**
   - 문제: 각 라벨이 독립적으로 샘플링 → 중복 발생
   - 해결: 사전에 세션 할당 후 각 라벨에 고유 prefix 파일 제공

4. **복구 전략**
   - 문제: 부분 실패 시 처음부터 재생성
   - 해결: Part 분할 + 병합 전략

### Best Practices

✅ **할 것**:
- 세션 할당을 사전에 명확히 분리
- 병렬 실행은 API limit 고려 (3-4개 권장)
- Progress 로그 자주 출력 (20개마다)
- 각 Part는 독립적으로 저장
- 최종 병합 전에 검증

❌ **하지 말 것**:
- 실행 중인 터미널에서 새 명령 실행
- 6개 이상 병렬 프로세스
- 같은 pool에서 각 라벨이 독립 샘플링
- 부분 실패 시 전체 재생성

---

## 📂 파일 구조

### 입력 파일
```
ESConv_normal_prefixes.json       # 1225 일반 세션
ESConv_v5_prefixes.json           # 75 위기 세션
```

### 할당 파일 (Phase 7.2 생성)
```
ESConv_v1_assigned.json           # 200 sessions for V1
ESConv_v2_assigned.json           # 200 sessions for V2
ESConv_v3_assigned.json           # 200 sessions for V3
ESConv_v4_assigned.json           # 100 sessions for V4
ESConv_normal_assigned.json       # 525 sessions for Normal
ESConv_v5_assigned.json           # 75 sessions for V5
session_assignments.json          # Assignment record
```

### 생성 스크립트
```
gen_v1.py                         # V1 generator
gen_v2.py                         # V2 generator
gen_v3.py                         # V3 generator
gen_v4.py                         # V4 generator
gen_normal_part1.py               # Normal part 1 (0-175)
gen_normal_part2.py               # Normal part 2 (175-350)
gen_normal_part3.py               # Normal part 3 (350-525)
```

### 출력 파일 (예정)
```
generated_V1.json                 # 200 samples
generated_V2.json                 # 200 samples
generated_V3.json                 # 200 samples
generated_V4.json                 # 100 samples
generated_Normal_part1.json       # 175 samples
generated_Normal_part2.json       # 175 samples
generated_Normal_part3.json       # 175 samples
generated_Normal.json             # 525 samples (merged)
```

### 검증 스크립트 (작성 예정)
```
merge_final_normal.py             # Normal parts merger
verify_final_dataset.py           # Final validation
split_train_test.py               # Train/Test split
```

---

## 🎓 요약

**현재 위치**: Phase 7.3 완료, Phase 8 대기 중

**준비 완료**:
- ✅ 1300개 유니크 세션 할당
- ✅ 7개 생성 스크립트 준비
- ✅ 중복 없음 검증 완료

**다음 액션**:
1. 1차 배치 실행 (V1, V2, V3)
2. 2차 배치 실행 (V4, Normal_part1, Normal_part2)
3. 3차 배치 실행 (Normal_part3)
4. Normal 병합
5. 최종 검증
6. Train/Test 분할

**예상 완료**: 2-3시간 내

---

**작성자**: GitHub Copilot  
**검토**: 필요 시 업데이트
