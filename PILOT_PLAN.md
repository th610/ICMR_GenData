# ESConv Violation Detector - Pilot Phase

## 🎯 파일럿 목표

**Small-scale validation (각 클래스 5개)**
- 프롬프트 품질 검증
- 데이터 생성 파이프라인 테스트
- Session-level 설계 검증
- 이후 전체 스케일(600개)로 확장

---

## 📊 파일럿 데이터 구성

```
총 30개 세션 (Single-label)

├─ Normal: 5
├─ V1: 5
├─ V2: 5
├─ V3: 5
├─ V4: 5
└─ V5: 5
```

### Split (Session-level)
- Train: 24 (각 클래스 4개)
- Val: 3 (각 클래스 0-1개)
- Test: 3 (각 클래스 0-1개)

---

## 🔧 파일럿 파이프라인

### Step 1: Normal 세션 준비
```bash
python scripts/pilot/step1_prepare_normal.py --num_sessions 5
```

**작업:**
- ESConv에서 5개 세션 샘플링
- 15~20턴으로 cut
- `data/pilot/normal/` 저장

### Step 2: V1~V3 세션 생성
```bash
python scripts/pilot/step2_generate_v1_v3.py --num_per_class 5
```

**작업:**
- Normal 5개 중 일부를 베이스로 사용
- 각 위반 타입별 5개씩 생성
- 마지막 Supporter 턴만 재작성
- `data/pilot/v1/`, `v2/`, `v3/` 저장

### Step 3: V4/V5 세션 생성
```bash
python scripts/pilot/step3_generate_v4_v5.py --num_per_class 5
```

**작업:**
- 완전 신규 멀티턴 생성 (10~15턴)
- Seeker + Supporter 모두 생성
- `data/pilot/v4/`, `v5/` 저장

### Step 4: 품질 검증 (수동)
```bash
python scripts/pilot/step4_manual_review.py
```

**작업:**
- 각 클래스 샘플 출력
- 사람이 검토:
  - 위반이 명확한가?
  - 자연스러운가?
  - 다른 위반과 섞이지 않았나?
- 문제 샘플 재생성

### Step 5: Session-level Split
```bash
python scripts/pilot/step5_split_sessions.py
```

**작업:**
- 30개 세션 → train/val/test (24/3/3)
- Session-level split (같은 세션 분리 안됨)

### Step 6: 학습 입력 생성
```bash
python scripts/pilot/step6_create_inputs.py
```

**작업:**
- 각 세션마다:
  - Prefix Summary (동적 생성, 최대 6 bullets)
  - Last 4 turns (원문)
  - Target response
- 토큰 길이 확인 (512 이내?)

### Step 7: Judge QA (선택)
```bash
python scripts/pilot/step7_judge_qa.py --sample_ratio 0.5
```

**작업:**
- 30개 중 15개 (50%)를 Judge로 재평가
- Agreement 확인
- 80% 미만이면 프롬프트 수정

### Step 8: 모델 학습
```bash
python scripts/pilot/step8_train.py
```

**작업:**
- Single-label classifier (6-class)
- Train: 24, Val: 3
- Macro F1 계산

---

## 💻 코드 구조 (파일럿)

### 기존 코드 활용 vs 새로 작성

**전략: 기존 코드 부분 수정 + 새 파일 추가**

```
generator_dataset/
├── configs/
│   ├── poc.yaml (기존, 참고용)
│   └── pilot.yaml (신규)
│
├── data/
│   └── pilot/
│       ├── normal/
│       ├── v1/
│       ├── v2/
│       ├── v3/
│       ├── v4/
│       ├── v5/
│       └── processed/
│           ├── train.jsonl
│           ├── val.jsonl
│           └── test.jsonl
│
├── scripts/
│   └── pilot/
│       ├── step1_prepare_normal.py
│       ├── step2_generate_v1_v3.py
│       ├── step3_generate_v4_v5.py
│       ├── step4_manual_review.py
│       ├── step5_split_sessions.py
│       ├── step6_create_inputs.py
│       ├── step7_judge_qa.py
│       └── step8_train.py
│
└── src/
    ├── llm/
    │   ├── openai_client.py (기존 재사용)
    │   ├── prompts_v2.py (신규, 새 프롬프트)
    │   └── judge_v2.py (신규)
    │
    ├── generation/
    │   ├── normal_cutter.py (신규)
    │   ├── v1_v3_rewriter.py (신규)
    │   └── v4_v5_generator.py (신규)
    │
    ├── preprocessing/
    │   ├── dynamic_summary.py (신규)
    │   └── input_builder.py (신규)
    │
    └── training/
        └── session_classifier.py (신규)
```

---

## 📝 configs/pilot.yaml

```yaml
# Pilot configuration
pilot:
  num_sessions_per_class: 5
  total_sessions: 30

# Normal session preparation
normal:
  source: "ESConv.json"
  target_length: [15, 20]  # Random cut
  num_sessions: 5

# V1~V3 generation (ESConv-based rewrite)
v1_v3:
  num_per_class: 5
  base_sessions: "use_from_normal"  # or "sample_new"
  rewrite_position: "last_supporter_turn"

# V4/V5 generation (Full multiturn)
v4_v5:
  num_per_class: 5
  session_length: [10, 15]
  violation_position: "last_supporter_turn"

# Summary settings
summary:
  max_bullets: 6
  max_tokens: 150
  summary_scope: "all_before_target"  # 동적 생성

# Input structure
input:
  max_length: 512
  structure:
    - prefix_summary
    - last_4_turns
    - target_response

# Split
split:
  train: 0.8   # 24
  val: 0.1    # 3
  test: 0.1   # 3

# LLM
llm:
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens_rewrite: 300
  max_tokens_summary: 200
  max_tokens_multiturn: 1500
  max_tokens_judge: 300

# Training
training:
  model_name: "distilroberta-base"
  num_classes: 6  # Normal + V1~V5
  batch_size: 4   # 작은 데이터셋
  learning_rate: 2e-5
  num_epochs: 10  # 작은 데이터라 epoch 많이
  warmup_steps: 10
```

---

## 🚀 실행 계획

### Week 1: 파일럿 구현 및 검증

**Day 1: 코드 작성**
- [ ] `pilot.yaml` 작성
- [ ] `prompts_v2.py` 작성 (새 프롬프트)
- [ ] `normal_cutter.py`, `v1_v3_rewriter.py`, `v4_v5_generator.py`
- [ ] `dynamic_summary.py`, `input_builder.py`

**Day 2: 데이터 생성**
- [ ] Step 1-3 실행 (30개 세션 생성)
- [ ] Step 4 수동 검증
- [ ] 문제 있으면 재생성

**Day 3: 학습 및 평가**
- [ ] Step 5-6 실행
- [ ] Step 8 학습
- [ ] 결과 분석

**Day 4-5: 반복 개선**
- [ ] 프롬프트 튜닝
- [ ] 품질 개선
- [ ] Judge 신뢰도 확인

### Week 2: 전체 스케일 확장 (파일럿 통과시)

**확장 계획:**
```yaml
# configs/full.yaml
full:
  normal: 300
  v1: 84
  v2: 83
  v3: 83
  v4: 25
  v5: 25
  total: 600
```

---

## ✅ 파일럿 성공 기준

1. **생성 품질**
   - 각 클래스 위반이 명확하게 구분됨
   - 자연스러운 대화 흐름
   - Judge agreement > 80%

2. **기술적 검증**
   - 데이터 누수 없음 (동적 summary 확인)
   - 토큰 길이 512 이내
   - 학습 파이프라인 정상 작동

3. **모델 성능**
   - Val Macro F1 > 0.4 (작은 데이터셋이라 낮아도 OK)
   - 각 클래스 1개 이상 맞춤
   - Confusion matrix 확인

**성공하면 → 전체 600개로 확장**
**실패하면 → 프롬프트/설계 수정 후 재시도**

---

## 💡 파일럿의 장점

1. **리스크 최소화**
   - 30개만 생성 → 비용 $0.02
   - 빠른 검증 (~2-3일)

2. **프롬프트 튜닝**
   - 소량으로 빠르게 반복
   - 품질 확인 후 확장

3. **설계 검증**
   - Session-level 구조 테스트
   - 동적 summary 검증
   - 토큰 길이 확인

4. **코드 안정화**
   - 버그 수정
   - 파이프라인 최적화
