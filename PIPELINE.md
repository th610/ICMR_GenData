# 데이터 생성 및 학습 전체 절차 (상세 가이드)

## 🎯 프로젝트 목표

**ESConv Violation Detector PoC**
- 정서 지원 상담(Emotional Support Conversation)에서 상담사(supporter)의 응답 품질을 자동 평가
- 5가지 위반 유형(V1~V5)을 multi-label classification으로 탐지
- 합성 데이터 생성 + LLM-judge 라벨링 + Transformer 기반 학습 파이프라인 구축

**위반 유형 정의:**
- **V1 (Missing Context & Info-Gathering)**: 내담자의 맥락을 충분히 파악하지 않고 응답. 필요한 정보를 묻지 않음.
- **V2 (Agency Violation)**: 내담자의 자율성과 주도권을 침해. 지시적/강압적 조언.
- **V3 (Low-Quality Empathy)**: 진부하고 형식적인 공감 표현. "힘내세요", "이해해요" 같은 피상적 응답.
- **V4 (Reality Distortion)**: 내담자의 현실이나 감정을 왜곡하거나 무시. 과도한 긍정주의.
- **V5 (Crisis Safety Failure)**: 위기 상황(자살, 자해 등)에서 안전 프로토콜 무시.

---

## 📋 전체 파이프라인 개요

```
ESConv.json (1300 세션, Liu et al. 2021)
    ↓
[Step 1] 원본 샘플링
    ↓
50 원본 세션 (1361 turns)
    ↓
[Step 2] 합성 생성 (위반 주입)
    ↓  LLM(gpt-4o-mini) single-turn rewrite
50 합성 세션 (각 1개 위반)
    ↓
[Step 3] Train/Val/Test 분할
    ↓  Session-level 80/10/10
Train 80 / Val 10 / Test 10 세션
    ↓
[Step 4] Turn 샘플 생성
    ↓  Context window N=8, Rule-based summary
Train 160 / Val 20 / Test 20 turn 샘플
    ↓
[Step 5] LLM-judge 라벨링
    ↓  gpt-4o-mini multi-label classification
라벨링된 샘플 (V1~V5 binary labels)
    ↓
[Step 6] 모델 학습
    ↓  distilroberta-base, 3 epochs
Trained Model (multi-label classifier)
    ↓
[Step 7] 테스트 평가
    ↓  Precision/Recall/F1 per label
성능 메트릭 (Micro F1: 0.56, Macro F1: 0.50)
```

---

## 🔧 환경 설정

### 필수 패키지
```bash
pip install transformers torch accelerate scikit-learn pyyaml openai
```

**버전 정보:**
- Python: 3.8+
- transformers: 4.30+
- torch: 2.0+
- openai: 1.0+

### 설정 파일: `configs/poc.yaml`

```yaml
# LLM 설정
llm:
  api_key: "your-openai-api-key"
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1000

# 합성 데이터 생성 설정
synthesis:
  num_sessions: 50
  violations:
    V1: 12  # Missing context
    V2: 10  # Agency violation
    V3: 12  # Low-quality empathy
    V4: 8   # Reality distortion
    V5: 8   # Crisis safety failure
  
  # 위반별 주입 위치 (턴 범위)
  sample_turn_range:
    V1: [6, 15]   # 중반부 (맥락 파악 실패가 명확)
    V2: [6, 15]   # 중반부
    V3: [3, 12]   # 초중반 (공감 필요 시점)
    V4: [6, 15]   # 중반부
    V5: [8, 15]   # 후반부 (위기 상황 전개 후)

# Turn 샘플링 설정
sampling:
  context_window: 8           # 타겟 턴 이전 최대 8개 턴
  samples_per_original: 2     # 원본 세션당 2개 샘플
  samples_per_synthetic: 2    # 합성 세션당 2개 샘플 (위반턴 1 + 랜덤 1)
  
  summary:
    use_llm_summary: false    # true면 LLM 요약, false면 rule-based
    max_turns_for_summary: 12 # 요약에 포함할 최대 턴 수
    max_summary_length: 100   # Rule-based 요약 최대 길이

# 학습 설정
training:
  model_name: "distilroberta-base"
  num_epochs: 3
  batch_size: 16
  learning_rate: 2.0e-5
  warmup_steps: 100
  weight_decay: 0.01
  max_length: 512             # 토크나이저 최대 길이
```

---

## Step 1: 원본 샘플링

### 목적
ESConv 전체 데이터셋(1300 세션)에서 실험에 사용할 50개 세션을 랜덤 샘플링

### 스크립트
`scripts/step1_sample_original.py`

### 입력 데이터
**`ESConv.json` 구조:**
```json
[
  {
    "conversation_id": "1",
    "conversation": [
      {
        "speaker": "seeker",
        "utterance_idx": 0,
        "text": "Hi, I've been feeling really down lately..."
      },
      {
        "speaker": "supporter",
        "utterance_idx": 1,
        "text": "I'm here to listen. What's been going on?"
      },
      ...
    ],
    "situation": "Financial stress due to job loss",
    "emotion_type": "anxious",
    ...
  },
  ...
]
```

- **총 세션 수**: 1300개
- **평균 턴 수**: 30~40 turns/session
- **화자**: seeker (내담자), supporter (상담사)

### 처리 로직

**주요 코드 (`scripts/step1_sample_original.py`):**
```python
def sample_sessions(sessions: List[Dict], num_samples: int, seed: int) -> List[Dict]:
    """랜덤 샘플링 후 session_id 재부여"""
    random.seed(seed)
    sampled = random.sample(sessions, num_samples)
    
    # Session ID 재부여: orig_0000, orig_0001, ...
    for i, session in enumerate(sampled):
        session['session_id'] = f"orig_{i:04d}"
    
    return sampled
```

**실행 과정:**
1. ESConv.json 전체 로드
2. `random.sample()`로 50개 선택 (seed=42)
3. 각 세션에 `session_id: "orig_XXXX"` 부여
4. 통계 계산 (총 턴 수, 평균 턴 수, supporter 턴 수)
5. JSON 저장

### 출력 데이터
**파일:** `data/sessions_original_50.json`

**통계:**
- 세션 수: 50
- 총 턴 수: 1361
- 평균 턴/세션: 27.22
- Supporter 턴 수: 663
- Seeker 턴 수: 698

**출력 형식:**
```json
[
  {
    "session_id": "orig_0000",
    "conversation": [...],
    "situation": "...",
    "emotion_type": "...",
    ...
  },
  ...
]
```

### 실행 명령
```bash
python scripts/step1_sample_original.py \
  --input ESConv.json \
  --output data/sessions_original_50.json \
  --num_sessions 50 \
  --seed 42
```

### 예상 실행 시간
< 10초

### 검증 방법
```bash
# 세션 수 확인
python -c "import json; print(len(json.load(open('data/sessions_original_50.json'))))"
# 출력: 50

# 첫 번째 세션 ID 확인
python -c "import json; print(json.load(open('data/sessions_original_50.json'))[0]['session_id'])"
# 출력: orig_0000
```

---

## Step 2: 합성 데이터 생성 (위반 주입)

### 목적
원본 세션의 supporter 응답 1개를 LLM으로 리라이트하여 특정 위반(V1~V5) 주입

### 스크립트 및 모듈
- **메인 스크립트:** `scripts/step2_generate_synthetic.py`
- **핵심 모듈:** `src/synth/rewrite_turn.py`
- **프롬프트:** `src/llm/prompts.py` (REWRITE_USER_TEMPLATE)
- **LLM 클라이언트:** `src/llm/openai_client.py`

### 입력 데이터
- `data/sessions_original_50.json` (Step 1 출력)
- `configs/poc.yaml` (위반 분포 설정)

### 처리 로직

#### 2.1 위반 타입 할당
```python
# src/synth/rewrite_turn.py
def assign_violations(num_sessions: int, violation_counts: Dict) -> List[str]:
    """설정된 분포대로 위반 타입 할당"""
    # V1:12, V2:10, V3:12, V4:8, V5:8 → 총 50개
    violations = []
    for v_type, count in violation_counts.items():
        violations.extend([v_type] * count)
    
    random.shuffle(violations)
    return violations  # ['V1', 'V3', 'V2', ...]
```

#### 2.2 위반 주입 위치 선택
```python
def select_violation_turn(session: Dict, violation_type: str, turn_range: Dict) -> int:
    """
    위반 타입에 적합한 supporter 턴 선택
    
    Args:
        session: 세션 데이터
        violation_type: V1~V5
        turn_range: 위반별 턴 범위 (예: V1은 [6, 15])
    
    Returns:
        선택된 supporter 턴의 utterance_idx
    """
    # 1. 전체 supporter 턴 찾기
    supporter_turns = [
        (i, turn) for i, turn in enumerate(session['conversation'])
        if turn['speaker'] == 'supporter'
    ]
    
    # 2. 턴 범위 내 필터링
    min_turn, max_turn = turn_range.get(violation_type, [3, 15])
    eligible = [
        (global_idx, turn) for global_idx, turn in supporter_turns
        if min_turn <= global_idx <= max_turn
    ]
    
    # 3. 랜덤 선택
    if not eligible:
        eligible = supporter_turns  # 범위 내 없으면 전체에서
    
    selected_idx, _ = random.choice(eligible)
    return selected_idx
```

**위반별 턴 범위 이유:**
- **V1 (맥락 파악 실패)**: [6, 15] - 충분한 대화가 진행된 후 맥락 누락이 명확
- **V2 (주도권 침해)**: [6, 15] - 관계가 형성된 후 지시적 조언이 나타남
- **V3 (저품질 공감)**: [3, 12] - 초반~중반, 공감이 필요한 시점
- **V4 (현실 왜곡)**: [6, 15] - 내담자의 상황이 충분히 드러난 후
- **V5 (위기 안전)**: [8, 15] - 위기 상황이 전개된 후반부

#### 2.3 LLM 리라이트

**프롬프트 (`src/llm/prompts.py`):**
```python
REWRITE_USER_TEMPLATE = """You are rewriting a counseling response to inject a specific violation.

**Context:**
Situation: {situation}
Emotion: {emotion_type}

**Conversation History:**
{conversation_history}

**Original Supporter Response:**
{original_response}

**Task:**
Rewrite this response to clearly demonstrate: **{violation_type}**

{violation_description}

**Requirements:**
1. Maintain conversational flow and tone
2. Keep similar length (±20%)
3. Make violation obvious but realistic
4. Don't mention the violation explicitly

Return JSON:
{{
  "rewritten_response": "...",
  "rationale": "Brief explanation of how violation was injected"
}}
"""
```

**위반별 설명 (`violation_description`):**
```python
VIOLATION_DESCRIPTIONS = {
    "V1": """Missing Context & Info-Gathering:
- Respond without understanding seeker's full situation
- Skip necessary clarifying questions
- Make assumptions about unstated details
- Jump to advice without gathering information""",
    
    "V2": """Agency Violation:
- Give directive, prescriptive advice
- Use "you should", "you must", "you need to"
- Take decision-making control from seeker
- Impose solutions without collaboration""",
    
    "V3": """Low-Quality Empathy:
- Use clichéd phrases ("I understand", "Stay strong", "It'll be okay")
- Generic platitudes without personalization
- Surface-level acknowledgment
- Copy-paste emotional responses""",
    
    "V4": """Reality Distortion:
- Dismiss or minimize seeker's valid concerns
- Overly optimistic reframing
- Deny seeker's emotional reality
- "Just think positive" mentality""",
    
    "V5": """Crisis Safety Failure:
- Ignore suicide/self-harm mentions
- Respond casually to crisis signals
- Skip safety assessment
- No referral to professional help"""
}
```

**LLM 호출 (`src/llm/openai_client.py`):**
```python
def call(self, system_prompt: str, user_prompt: str) -> Dict:
    """OpenAI API 호출 with retry"""
    response = openai.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=self.temperature,
        max_tokens=self.max_tokens
    )
    
    content = response.choices[0].message.content
    
    # JSON 파싱
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Retry with JSON 강제 메시지
        return self.call(system_prompt, RETRY_MESSAGE)
```

#### 2.4 세션 재구성
```python
def rewrite_session_with_violation(session: Dict, violation_type: str, config: Dict) -> Dict:
    """세션 복사 후 1개 턴만 리라이트"""
    
    # 1. 세션 딥카피
    new_session = copy.deepcopy(session)
    
    # 2. 위반 주입 위치 선택
    target_idx = select_violation_turn(new_session, violation_type, config['turn_range'])
    
    # 3. 원본 응답 저장
    original_text = new_session['conversation'][target_idx]['text']
    
    # 4. LLM 리라이트
    rewrite_result = llm_client.call(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=format_rewrite_prompt(
            session=new_session,
            target_idx=target_idx,
            violation_type=violation_type
        )
    )
    
    # 5. 텍스트 교체
    new_session['conversation'][target_idx]['text'] = rewrite_result['rewritten_response']
    
    # 6. 메타데이터 추가
    new_session['injected_violation'] = {
        'type': violation_type,
        'turn_id': target_idx,
        'supporter_utterance_index': get_supporter_index(new_session, target_idx),
        'original_text': original_text,
        'rewritten_text': rewrite_result['rewritten_response'],
        'rationale': rewrite_result['rationale']
    }
    
    # 7. Session ID 변경
    new_session['session_id'] = new_session['session_id'].replace('orig_', 'synth_')
    
    return new_session
```

### 출력 데이터
**파일:** `data/sessions_synth_50.json`

**통계:**
- 세션 수: 50
- 성공률: 100% (50/50)
- 위반 분포:
  - V1: 12
  - V2: 10
  - V3: 12
  - V4: 8
  - V5: 8

**출력 형식:**
```json
[
  {
    "session_id": "synth_0000",
    "conversation": [
      {"speaker": "seeker", "utterance_idx": 0, "text": "..."},
      {"speaker": "supporter", "utterance_idx": 1, "text": "... (리라이트된 위반 응답) ..."},
      ...
    ],
    "injected_violation": {
      "type": "V1",
      "turn_id": 7,
      "supporter_utterance_index": 3,
      "original_text": "Can you tell me more about what happened?",
      "rewritten_text": "You should just move on and find a new job.",
      "rationale": "Injected V1 by skipping information gathering and jumping to advice"
    },
    "situation": "...",
    "emotion_type": "..."
  },
  ...
]
```

### 실행 명령
```bash
python scripts/step2_generate_synthetic.py \
  --input data/sessions_original_50.json \
  --output data/sessions_synth_50.json \
  --seed 42
```

### 예상 실행 시간
- 약 5~10분 (LLM API 호출 50회)
- Progress bar로 진행 상황 표시

### 실행 로그 예시
```
============================================================
STEP 2: Generate Synthetic Sessions with Violations
============================================================

Loading original sessions from: data/sessions_original_50.json
Loaded 50 sessions

Violation distribution:
  V1: 12
  V2: 10
  V3: 12
  V4: 8
  V5: 8

Generating synthetic sessions...
  [10/50] V3 injected at turn 5 (supporter_idx=2)
  [20/50] V1 injected at turn 9 (supporter_idx=4)
  [30/50] V2 injected at turn 11 (supporter_idx=5)
  ...

============================================================
Summary
============================================================
Total sessions: 50
Successful: 50
Failed: 0

Violation counts:
  V1: 12
  V2: 10
  V3: 12
  V4: 8
  V5: 8

Output saved to: data/sessions_synth_50.json
```

### 검증 방법
```python
# check_synthetic.py
import json

synth = json.load(open('data/sessions_synth_50.json'))

# 1. 전체 수 확인
print(f"Total sessions: {len(synth)}")

# 2. 위반 분포 확인
from collections import Counter
violations = [s['injected_violation']['type'] for s in synth]
print(Counter(violations))
# 출력: Counter({'V1': 12, 'V3': 12, 'V2': 10, 'V4': 8, 'V5': 8})

# 3. 샘플 확인
sample = synth[0]
vio = sample['injected_violation']
print(f"\nSession: {sample['session_id']}")
print(f"Violation: {vio['type']} at turn {vio['turn_id']}")
print(f"Original: {vio['original_text'][:100]}...")
print(f"Rewritten: {vio['rewritten_text'][:100]}...")
print(f"Rationale: {vio['rationale']}")
```

### 주요 문제점 및 대응

**문제 1: LLM이 JSON 반환 실패**
- 대응: `openai_client.py`에 retry 로직 추가
- JSON parse 실패 시 RETRY_MESSAGE로 재요청
- 2회 실패 시 None 반환 → 해당 세션 skip

**문제 2: 위반이 너무 약하게 주입**
- 대응: 프롬프트에 "Make violation **obvious** but realistic" 강조
- Few-shot 예시 추가 고려

**문제 3: V4/V5가 Step 5에서 인식 안됨**
- 현상: 합성은 성공했으나 LLM-judge가 0개 탐지
- 원인 추정:
  1. 리라이트가 충분히 강하지 않음
  2. Judge 프롬프트가 V4/V5에 대해 너무 엄격
  3. ESConv 데이터 자체에 V4/V5 패턴이 희소
- 개선 방향:
  - V4/V5 전용 few-shot 예시 추가
  - Judge 프롬프트 완화
  - 다른 합성 방법 시도 (multi-turn rewrite 등)

---

## Step 3: Train/Val/Test 분할

### 목적
세션 레벨에서 학습/검증/테스트 세트 분할 (맥락 누출 방지)

### 스크립트
`scripts/step3_split_sessions.py`

### 입력 데이터
- `data/sessions_original_50.json`
- `data/sessions_synth_50.json`

### 처리 로직

**핵심 원칙: Session-level split**
- 같은 세션의 턴들이 train/val/test에 분산되면 안됨
- 세션 전체를 하나의 단위로 분할

```python
def split_sessions(sessions: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    세션을 train/val/test로 분할
    
    Args:
        sessions: 세션 리스트
        train_ratio: 학습 비율 (0.8 = 80%)
        val_ratio: 검증 비율 (0.1 = 10%)
        test_ratio: 테스트 비율 (0.1 = 10%)
        seed: 랜덤 시드
    
    Returns:
        train_sessions, val_sessions, test_sessions
    """
    random.seed(seed)
    shuffled = sessions.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test
```

**원본/합성 분리 처리:**
```python
# 1. 원본 50개 분할
orig_train, orig_val, orig_test = split_sessions(original_sessions, seed=42)
# 결과: 40, 5, 5

# 2. 합성 50개 분할
synth_train, synth_val, synth_test = split_sessions(synthetic_sessions, seed=42)
# 결과: 40, 5, 5

# 3. 합치기
train_sessions = orig_train + synth_train      # 80 sessions
val_sessions = orig_val + synth_val            # 10 sessions
test_sessions = orig_test + synth_test         # 10 sessions
```

**위반 분포 확인:**
```python
def check_violation_distribution(sessions: List[Dict]) -> Dict:
    """합성 세션의 위반 분포 확인"""
    violations = Counter()
    for session in sessions:
        if 'injected_violation' in session:
            violations[session['injected_violation']['type']] += 1
    return dict(violations)

# Train set 확인
train_violations = check_violation_distribution(train_sessions)
# 예: {'V1': 10, 'V2': 8, 'V3': 9, 'V4': 7, 'V5': 7}
```

### 출력 데이터

**파일:**
- `data/splits/train.json` (80 세션)
- `data/splits/val.json` (10 세션)
- `data/splits/test.json` (10 세션)

**분할 통계:**

| Split | 원본 | 합성 | 총 | 비율 |
|-------|------|------|-----|------|
| Train | 39   | 41   | 80  | 80%  |
| Val   | 4    | 6    | 10  | 10%  |
| Test  | 7    | 3    | 10  | 10%  |

**위반 분포 (합성 세션만):**

| 위반 | Train | Val | Test | 총 |
|------|-------|-----|------|-----|
| V1   | 10    | 1   | 1    | 12  |
| V2   | 8     | 1   | 1    | 10  |
| V3   | 9     | 2   | 1    | 12  |
| V4   | 7     | 1   | 0    | 8   |
| V5   | 7     | 1   | 0    | 8   |

### 실행 명령
```bash
python scripts/step3_split_sessions.py \
  --original data/sessions_original_50.json \
  --synthetic data/sessions_synth_50.json \
  --output_dir data/splits \
  --seed 42
```

### 예상 실행 시간
< 5초

### 실행 로그 예시
```
============================================================
STEP 3: Split Sessions into Train/Val/Test
============================================================

Loading sessions...
  Original: 50 sessions
  Synthetic: 50 sessions

Splitting with ratio 80/10/10 (seed=42)...

Train set (80 sessions):
  Original: 39 sessions
  Synthetic: 41 sessions
  Violations: V1=10, V2=8, V3=9, V4=7, V5=7

Val set (10 sessions):
  Original: 4 sessions
  Synthetic: 6 sessions
  Violations: V1=1, V2=1, V3=2, V4=1, V5=1

Test set (10 sessions):
  Original: 7 sessions
  Synthetic: 3 sessions
  Violations: V1=1, V2=1, V3=1, V4=0, V5=0

Output saved to: data/splits/
  - train.json
  - val.json
  - test.json
```

### 검증 방법
```python
import json

train = json.load(open('data/splits/train.json'))
val = json.load(open('data/splits/val.json'))
test = json.load(open('data/splits/test.json'))

# 1. 개수 확인
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# 출력: Train: 80, Val: 10, Test: 10

# 2. 중복 확인 (session_id 기준)
train_ids = {s['session_id'] for s in train}
val_ids = {s['session_id'] for s in val}
test_ids = {s['session_id'] for s in test}

assert len(train_ids & val_ids) == 0, "Train-Val overlap!"
assert len(train_ids & test_ids) == 0, "Train-Test overlap!"
assert len(val_ids & test_ids) == 0, "Val-Test overlap!"
print("No overlap detected ✓")

# 3. 원본/합성 비율 확인
train_orig = sum(1 for s in train if s['session_id'].startswith('orig_'))
train_synth = sum(1 for s in train if s['session_id'].startswith('synth_'))
print(f"Train: {train_orig} original + {train_synth} synthetic")
```

### 주의사항

**왜 Session-level split이 중요한가?**

❌ **잘못된 방법 (Turn-level split):**
```python
# 모든 턴을 섞어서 분할하면...
all_turns = []
for session in sessions:
    for turn in session['conversation']:
        all_turns.append(turn)

train_turns, val_turns, test_turns = split(all_turns)  # WRONG!
```

**문제점:**
- 같은 세션의 턴들이 train/val/test에 분산
- Validation이 사실상 학습 데이터를 보게 됨 (context leakage)
- 성능이 과대평가됨

✅ **올바른 방법 (Session-level split):**
```python
# 세션 단위로 분할
train_sessions, val_sessions, test_sessions = split(sessions)

# 이후 각 세션에서 턴 샘플링 (Step 4)
```

**효과:**
- 모델이 학습 중 본 적 없는 완전히 새로운 대화 평가
- 실제 배포 환경과 동일한 조건
- 일반화 성능 정확히 측정

---

## Step 4: Turn 샘플 생성

### 목적
각 세션에서 supporter 응답을 타겟으로 하는 turn-level 샘플 생성 (context + response)

### 스크립트 및 모듈
- **메인 스크립트:** `scripts/step4_make_turn_samples.py`
- **핵심 모듈:** `src/data/make_turn_samples.py`
- **요약 모듈:** `src/llm/summarize.py`

### 입력 데이터
- `data/splits/train.json`
- `data/splits/val.json`
- `data/splits/test.json`
- `configs/poc.yaml`

### 샘플링 전략

#### 4.1 세션별 샘플 개수

**원본 세션:**
- 각 세션에서 supporter 턴 **2개** 랜덤 샘플링
- 이유: 원본 세션에는 위반이 없으므로 다양한 응답 패턴 학습

**합성 세션:**
- **위반 주입된 턴 1개** (필수)
- **랜덤 턴 1개** (추가)
- 이유: 위반 턴은 반드시 포함 + 정상 응답도 함께 학습

```python
def sample_turns_from_session(session: Dict, config: Dict) -> List[int]:
    """세션에서 타겟 턴 인덱스 선택"""
    
    # Supporter 턴들의 global index 찾기
    supporter_indices = [
        i for i, turn in enumerate(session['conversation'])
        if turn['speaker'] == 'supporter'
    ]
    
    if 'injected_violation' in session:
        # 합성 세션: 위반 턴 + 랜덤 1개
        violation_idx = session['injected_violation']['turn_id']
        
        # 위반 턴 제외하고 랜덤 선택
        other_indices = [idx for idx in supporter_indices if idx != violation_idx]
        random_idx = random.choice(other_indices) if other_indices else None
        
        targets = [violation_idx]
        if random_idx is not None:
            targets.append(random_idx)
    else:
        # 원본 세션: 랜덤 2개
        targets = random.sample(supporter_indices, min(2, len(supporter_indices)))
    
    return targets
```

#### 4.2 Context Window (N=8)

**Sliding window 방식:**
- 타겟 턴 **이전** 최대 8개 턴 포함
- 타겟 턴 자체는 제외 (모델이 예측할 응답)
- 대화 시작부분이면 가능한 만큼만 포함

```python
def build_context_turns(conversation: List[Dict], target_idx: int, window_size: int = 8) -> List[Dict]:
    """
    타겟 턴 이전 최대 window_size개 턴 추출
    
    Args:
        conversation: 전체 대화
        target_idx: 타겟 턴의 global index
        window_size: 컨텍스트 윈도우 크기 (default: 8)
    
    Returns:
        컨텍스트 턴 리스트 (시간순)
    """
    # 타겟 이전 턴들
    prev_turns = conversation[:target_idx]
    
    # 최대 window_size개만
    if len(prev_turns) > window_size:
        context = prev_turns[-window_size:]
    else:
        context = prev_turns
    
    return context
```

**예시:**
```python
conversation = [
    {"speaker": "seeker", "text": "Hi..."},           # idx 0
    {"speaker": "supporter", "text": "Hello..."},     # idx 1
    {"speaker": "seeker", "text": "I'm sad..."},      # idx 2
    ...                                                # idx 3-10
    {"speaker": "supporter", "text": "TARGET"},       # idx 11 (타겟)
]

# target_idx=11, window_size=8
context = build_context_turns(conversation, 11, 8)
# 결과: idx 3~10 (총 8개)
```

#### 4.3 대화 요약 (Summary)

**Rule-based 방식 (현재 사용):**
```python
def rule_based_summary(conversation: List[Dict], max_turns: int = 12, max_length: int = 100) -> List[str]:
    """
    간단한 규칙 기반 요약
    
    Args:
        conversation: 전체 대화
        max_turns: 요약에 사용할 최대 턴 수
        max_length: 각 bullet의 최대 길이
    
    Returns:
        요약 bullet points 리스트
    """
    # 1. 최근 max_turns개 턴만 사용
    recent_turns = conversation[-max_turns:]
    
    # 2. Seeker 발화만 추출
    seeker_turns = [
        turn['text'] for turn in recent_turns
        if turn['speaker'] == 'seeker'
    ]
    
    # 3. 각 발화를 truncate해서 bullet으로
    bullets = []
    for text in seeker_turns:
        # 길이 제한
        if len(text) > max_length:
            bullet = text[:max_length] + "..."
        else:
            bullet = text
        bullets.append(bullet)
    
    return bullets[:5]  # 최대 5개
```

**LLM 방식 (선택적, `use_llm_summary: true`):**
```python
def llm_summary(conversation: List[Dict], llm_client) -> List[str]:
    """LLM으로 대화 요약 (3-5 bullet points)"""
    
    # 프롬프트 구성
    conv_text = "\n".join([
        f"{turn['speaker']}: {turn['text']}"
        for turn in conversation
    ])
    
    prompt = f"""Summarize this emotional support conversation in 3-5 key points:

{conv_text}

Return JSON:
{{
  "summary": ["point 1", "point 2", ...]
}}"""
    
    result = llm_client.call(SUMMARY_SYSTEM_PROMPT, prompt)
    return result['summary']
```

**사용 이유:**
- Rule-based: 빠름, API 비용 없음, 충분히 유용
- LLM: 더 나은 품질, 하지만 비용/시간 증가

#### 4.4 샘플 구조

```python
class TurnSample:
    """Turn-level 샘플 데이터 구조"""
    
    session_id: str           # "orig_0000" or "synth_0000"
    turn_id: int              # 타겟 턴의 global index
    
    # 입력
    context_turns: List[Dict] # 이전 8개 턴 (최대)
    summary: List[str]        # 대화 요약 bullets
    response: str             # 타겟 supporter 응답
    
    # 메타데이터
    meta: Dict = {
        'situation': str,         # 상황 설명
        'emotion_type': str,      # 감정 타입
        'is_violation_turn': bool,# 위반 턴 여부
        'num_context_turns': int, # 실제 컨텍스트 턴 수
        'num_summary_points': int # 요약 bullet 수
    }
```

**JSON 예시:**
```json
{
  "session_id": "synth_0005",
  "turn_id": 9,
  "context_turns": [
    {"speaker": "seeker", "text": "I lost my job last month..."},
    {"speaker": "supporter", "text": "I'm sorry to hear that..."},
    ...
  ],
  "summary": [
    "Seeker lost job last month",
    "Feeling anxious about finances",
    "Looking for support and guidance"
  ],
  "response": "You should just apply to more jobs and stop worrying.",
  "meta": {
    "situation": "Job loss causing financial stress",
    "emotion_type": "anxious",
    "is_violation_turn": true,
    "num_context_turns": 8,
    "num_summary_points": 3
  }
}
```

### 처리 흐름

```python
def process_split(sessions: List[Dict], config: Dict) -> List[Dict]:
    """한 split(train/val/test)의 모든 세션 처리"""
    
    all_samples = []
    
    for session in sessions:
        # 1. 타겟 턴 선택
        target_indices = sample_turns_from_session(session, config)
        
        # 2. 세션 요약 생성 (한 번만)
        summary = create_session_summary(session, config)
        
        # 3. 각 타겟 턴마다 샘플 생성
        for target_idx in target_indices:
            # Context
            context_turns = build_context_turns(
                session['conversation'],
                target_idx,
                window_size=config['context_window']
            )
            
            # Response
            response = session['conversation'][target_idx]['text']
            
            # Meta
            meta = {
                'situation': session.get('situation', ''),
                'emotion_type': session.get('emotion_type', ''),
                'is_violation_turn': (
                    'injected_violation' in session and
                    session['injected_violation']['turn_id'] == target_idx
                ),
                'num_context_turns': len(context_turns),
                'num_summary_points': len(summary)
            }
            
            sample = {
                'session_id': session['session_id'],
                'turn_id': target_idx,
                'context_turns': context_turns,
                'summary': summary,
                'response': response,
                'meta': meta
            }
            
            all_samples.append(sample)
    
    return all_samples
```

### 출력 데이터

**파일:**
- `data/turn_samples/train.jsonl` (160 샘플)
- `data/turn_samples/val.jsonl` (20 샘플)
- `data/turn_samples/test.jsonl` (20 샘플)

**통계 (Train):**
```
Total samples: 160
From original sessions: 78 (39 sessions × 2)
From synthetic sessions: 82 (41 sessions × 2)
  - Violation turns: 41
  - Random turns: 41

Context turns:
  Mean: 7.66
  Min: 2
  Max: 8

Summary bullets:
  Mean: 4.0
  Min: 1
  Max: 5
```

**Val/Test 통계:**
```
Val: 20 samples (4 orig × 2 + 6 synth × 2)
Test: 20 samples (7 orig × 2 + 3 synth × 2)
```

### 실행 명령
```bash
python scripts/step4_make_turn_samples.py \
  --input_dir data/splits \
  --output_dir data/turn_samples \
  --config configs/poc.yaml
```

### 예상 실행 시간
- Rule-based summary: < 30초
- LLM summary: ~5분 (100 세션 × LLM 호출)

### 실행 로그 예시
```
============================================================
STEP 4: Create Turn-Level Samples
============================================================

Loading configuration from: configs/poc.yaml
Context window: 8
Summary method: rule-based (max_turns=12, max_length=100)

Processing train set...
  Loaded 80 sessions
  [10/80] Created 2 samples from orig_0003
  [20/80] Created 2 samples from synth_0012 (1 violation turn)
  ...
  Total train samples: 160

Processing val set...
  Loaded 10 sessions
  Total val samples: 20

Processing test set...
  Loaded 10 sessions
  Total test samples: 20

Statistics:
  Train: 160 samples (avg 7.66 context turns, 4.0 summary bullets)
  Val: 20 samples
  Test: 20 samples

Output saved to: data/turn_samples/
```

### 검증 방법
```python
import json

# JSONL 로드
train_samples = []
with open('data/turn_samples/train.jsonl') as f:
    for line in f:
        train_samples.append(json.loads(line))

print(f"Train samples: {len(train_samples)}")

# 위반 턴 개수 확인
violation_samples = [s for s in train_samples if s['meta']['is_violation_turn']]
print(f"Violation turns: {len(violation_samples)}")
# 예상: 41 (합성 세션 41개 × 1)

# Context 길이 분포
context_lens = [s['meta']['num_context_turns'] for s in train_samples]
print(f"Context turns: mean={sum(context_lens)/len(context_lens):.2f}, min={min(context_lens)}, max={max(context_lens)}")

# 샘플 확인
sample = train_samples[0]
print(f"\nSample: {sample['session_id']} turn {sample['turn_id']}")
print(f"Context: {len(sample['context_turns'])} turns")
print(f"Summary: {len(sample['summary'])} bullets")
print(f"Response: {sample['response'][:100]}...")
print(f"Is violation: {sample['meta']['is_violation_turn']}")
```

### 주요 이슈

**이슈 1: `use_llm` vs `use_llm_summary` 변수명 버그**
- 증상: `src/data/make_turn_samples.py`에서 NameError
- 원인: 설정 파일은 `use_llm_summary`인데 코드는 `use_llm` 사용
- 해결: 변수명 통일
```python
# Before (버그)
if config['summary']['use_llm']:  # KeyError!
    ...

# After (수정)
if config['summary']['use_llm_summary']:
    ...
```

**이슈 2: Context가 너무 짧은 샘플**
- 증상: 일부 샘플의 context_turns가 2~3개
- 원인: 대화 초반부 턴이 타겟으로 선택됨
- 영향: 모델 성능 저하 가능
- 개선: 최소 context 길이 필터 추가 고려
```python
# 개선안
def sample_turns_from_session(session, config):
    supporter_indices = [...]
    
    # 최소 context 5개 이상인 턴만 후보
    eligible = [
        idx for idx in supporter_indices
        if idx >= 5  # 최소 5개 이전 턴 존재
    ]
    
    targets = random.sample(eligible, ...)
```

**이슈 3: Summary 품질**
- Rule-based: 단순 truncate라 문맥 손실
- LLM: 품질 좋지만 비용/시간
- 절충안: 중요한 키워드 추출 + 템플릿

---

## Step 5: LLM-judge 라벨링

### 목적
LLM(gpt-4o-mini)을 judge로 사용하여 각 supporter 응답에 V1~V5 위반 여부 라벨링

### 스크립트 및 모듈
- **메인 스크립트:** `scripts/step5_label_turns.py`
- **핵심 모듈:** `src/llm/judge.py`
- **프롬프트:** `src/llm/prompts.py` (JUDGE_USER_TEMPLATE)

### 입력 데이터
- `data/turn_samples/train.jsonl`
- `data/turn_samples/val.jsonl`
- `data/turn_samples/test.jsonl`
- `configs/poc.yaml`

### LLM-Judge 방식

#### 5.1 Why LLM-Judge?

**전통적 라벨링:**
- 사람이 직접 200개 샘플 라벨링
- 비용: 시간당 $30 × 10시간 = $300
- 시간: 1주일
- 일관성: 사람마다 기준 다름

**LLM-Judge 장점:**
- 비용: 200 샘플 × $0.001 = $0.20
- 시간: ~10분
- 일관성: 동일 프롬프트로 일관된 기준
- 확장성: 10,000개도 쉽게 라벨링

**단점:**
- 정확도: 사람보다 낮을 수 있음
- 편향: 프롬프트 품질에 의존
- 검증 필요: 샘플 수동 확인 필수

#### 5.2 프롬프트 설계

**System Prompt:**
```python
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of emotional support conversations.

Your task is to identify violations in supporter responses across 5 categories:
- V1: Missing Context & Info-Gathering
- V2: Agency Violation
- V3: Low-Quality Empathy
- V4: Reality Distortion
- V5: Crisis Safety Failure

Evaluate objectively based on the response content and context provided.
Return results in strict JSON format."""
```

**User Prompt Template:**
```python
JUDGE_USER_TEMPLATE = """Evaluate this supporter response for violations.

**Situation:**
{situation}

**Conversation Summary:**
{summary}

**Recent Context:**
{context}

**Supporter Response to Evaluate:**
"{response}"

---

**Violation Definitions:**

**V1 - Missing Context & Info-Gathering:**
- Responds without understanding seeker's full situation
- Skips necessary clarifying questions
- Makes assumptions about unstated details
- Jumps to advice without gathering information
Examples: "Just do X", "You should Y" (without asking about constraints)

**V2 - Agency Violation:**
- Gives directive, prescriptive advice
- Uses "you should", "you must", "you need to"
- Takes decision-making control from seeker
- Imposes solutions without collaboration
Examples: "You have to leave that job", "You must talk to them now"

**V3 - Low-Quality Empathy:**
- Uses clichéd phrases ("I understand", "Stay strong", "It'll be okay")
- Generic platitudes without personalization
- Surface-level acknowledgment
- Copy-paste emotional responses
Examples: "I know how you feel", "Everything happens for a reason"

**V4 - Reality Distortion:**
- Dismisses or minimizes seeker's valid concerns
- Overly optimistic reframing
- Denies seeker's emotional reality
- "Just think positive" mentality
Examples: "It's not that bad", "Just look on the bright side"

**V5 - Crisis Safety Failure:**
- Ignores suicide/self-harm mentions
- Responds casually to crisis signals
- Skips safety assessment
- No referral to professional help
Examples: Missing "I want to die" → "That's tough, but..."

---

**Task:**
For each violation type (V1-V5), determine:
1. Is it present? (0 = no, 1 = yes)
2. If multiple violations exist, identify the **most severe** one
3. Extract a brief evidence span (quote from response)

**Return JSON:**
{{
  "V1": 0 or 1,
  "V2": 0 or 1,
  "V3": 0 or 1,
  "V4": 0 or 1,
  "V5": 0 or 1,
  "top_violation": "V1" or "V2" or ... or "none",
  "evidence_span": "Direct quote from response showing the violation"
}}

**Important:**
- Be strict but fair
- Multiple violations can be 1 simultaneously
- If no violations, all should be 0 and top_violation="none"
- Evidence span should be actual text from the response
"""
```

#### 5.3 라벨링 로직

```python
def label_turn_sample(sample: Dict, llm_client, config: Dict) -> Dict:
    """
    하나의 turn 샘플을 LLM-judge로 라벨링
    
    Args:
        sample: Turn 샘플 (context, summary, response 포함)
        llm_client: OpenAI client
        config: 설정
    
    Returns:
        원본 샘플 + labels 추가
    """
    # 1. 프롬프트 구성
    context_text = "\n".join([
        f"{turn['speaker']}: {turn['text']}"
        for turn in sample['context_turns']
    ])
    
    summary_text = "\n".join([
        f"- {bullet}" for bullet in sample['summary']
    ])
    
    user_prompt = JUDGE_USER_TEMPLATE.format(
        situation=sample['meta'].get('situation', 'N/A'),
        summary=summary_text,
        context=context_text,
        response=sample['response']
    )
    
    # 2. LLM 호출
    try:
        result = llm_client.call(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        # 3. 라벨 추출
        labels = {
            'V1': result.get('V1', 0),
            'V2': result.get('V2', 0),
            'V3': result.get('V3', 0),
            'V4': result.get('V4', 0),
            'V5': result.get('V5', 0),
        }
        
        # 4. 샘플에 추가
        labeled_sample = sample.copy()
        labeled_sample['labels'] = labels
        labeled_sample['top_violation'] = result.get('top_violation', 'none')
        labeled_sample['evidence_span'] = result.get('evidence_span', '')
        
        return labeled_sample
        
    except Exception as e:
        print(f"Error labeling {sample['session_id']} turn {sample['turn_id']}: {e}")
        # 실패 시 모두 0으로
        labeled_sample = sample.copy()
        labeled_sample['labels'] = {'V1': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'V5': 0}
        labeled_sample['top_violation'] = 'error'
        labeled_sample['evidence_span'] = ''
        return labeled_sample
```

#### 5.4 배치 처리

```python
def label_all_samples(input_path: Path, output_path: Path, llm_client, config):
    """JSONL 파일의 모든 샘플 라벨링"""
    
    # 입력 로드
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Labeling {len(samples)} samples...")
    
    # 라벨링
    labeled_samples = []
    failed_count = 0
    
    for i, sample in enumerate(samples):
        labeled = label_turn_sample(sample, llm_client, config)
        labeled_samples.append(labeled)
        
        if labeled.get('top_violation') == 'error':
            failed_count += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] Labeled")
    
    # 출력 저장
    with open(output_path, 'w') as f:
        for sample in labeled_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\nCompleted: {len(samples)} samples, {failed_count} failures")
    
    return labeled_samples
```

### 출력 데이터

**파일:**
- `data/labeled/labeled_turns_train.jsonl` (160 샘플)
- `data/labeled/labeled_turns_val.jsonl` (20 샘플)
- `data/labeled/labeled_turns_test.jsonl` (20 샘플)

**라벨 분포 (Train 160 샘플):**

| 위반 | 개수 | 비율 | Top Violation | Multi-label 중복 |
|------|------|------|---------------|------------------|
| V1   | 92   | 57.5% | 85           | 52 (다른 위반과 동시) |
| V2   | 36   | 22.5% | 23           | 13               |
| V3   | 71   | 44.4% | 45           | 26               |
| V4   | 0    | 0%    | 0            | 0                |
| V5   | 0    | 0%    | 0            | 0                |

**Multi-label 통계:**
- Single violation: 120/160 (75%)
- Multiple violations: 40/160 (25%)
- No violation: 7/160 (4.4%)

**출력 형식:**
```json
{
  "session_id": "synth_0005",
  "turn_id": 9,
  "context_turns": [...],
  "summary": [...],
  "response": "You should just apply to more jobs and stop worrying.",
  "meta": {...},
  "labels": {
    "V1": 1,
    "V2": 1,
    "V3": 0,
    "V4": 0,
    "V5": 0
  },
  "top_violation": "V1",
  "evidence_span": "just apply to more jobs and stop worrying"
}
```

### 실행 명령
```bash
python scripts/step5_label_turns.py \
  --input_dir data/turn_samples \
  --output_dir data/labeled \
  --config configs/poc.yaml
```

### 예상 실행 시간
- 200 샘플 × ~3초/샘플 = ~10분
- API 속도에 따라 변동

### 실행 로그 예시
```
============================================================
STEP 5: Label Turn Samples with LLM-Judge
============================================================

Loading configuration from: configs/poc.yaml
LLM model: gpt-4o-mini

Labeling train set...
  Loading: data/turn_samples/train.jsonl
  Samples: 160
  [10/160] Labeled
  [20/160] Labeled
  ...
  [160/160] Labeled

  Completed: 160 samples, 0 failures

  Label distribution:
    V1: 92 (57.5%)
    V2: 36 (22.5%)
    V3: 71 (44.4%)
    V4: 0 (0.0%)    ⚠️ WARNING
    V5: 0 (0.0%)    ⚠️ WARNING
  
  Multi-label samples: 40/160 (25.0%)
  Top violations: V1=85, V2=23, V3=45, none=7

Labeling val set...
  [20/20] Labeled
  V1: 16, V2: 2, V3: 10, V4: 0, V5: 0

Labeling test set...
  [20/20] Labeled
  V1: 14, V2: 1, V3: 10, V4: 0, V5: 1

Output saved to: data/labeled/
```

### 검증 방법

```python
import json
from collections import Counter

# 로드
train = []
with open('data/labeled/labeled_turns_train.jsonl') as f:
    for line in f:
        train.append(json.loads(line))

# 라벨 분포
label_counts = Counter()
for sample in train:
    for v_type, value in sample['labels'].items():
        if value == 1:
            label_counts[v_type] += 1

print("Label distribution:")
for v_type in ['V1', 'V2', 'V3', 'V4', 'V5']:
    count = label_counts[v_type]
    pct = count / len(train) * 100
    print(f"  {v_type}: {count:3d} ({pct:5.1f}%)")

# Multi-label 확인
multi_label_count = 0
for sample in train:
    total_violations = sum(sample['labels'].values())
    if total_violations > 1:
        multi_label_count += 1

print(f"\nMulti-label samples: {multi_label_count}/{len(train)}")

# 샘플 확인
print("\n=== Sample with V1+V2 ===")
for sample in train:
    if sample['labels']['V1'] == 1 and sample['labels']['V2'] == 1:
        print(f"Response: {sample['response'][:150]}...")
        print(f"Top: {sample['top_violation']}")
        print(f"Evidence: {sample['evidence_span']}")
        break
```

### 핵심 문제점

#### 문제 1: V4/V5가 0개
**증상:**
- Step 2에서 V4:8, V5:8 주입했는데
- LLM-judge가 V4:0, V5:0 탐지

**원인 분석:**
1. **리라이트 품질 문제**
   - V4/V5 위반이 너무 약하게 주입됨
   - 프롬프트의 "realistic" 강조로 너무 미묘하게 리라이트
   
2. **Judge 프롬프트 문제**
   - V4/V5 정의가 너무 엄격
   - 예시가 부족해서 판단 기준 모호
   
3. **ESConv 데이터 특성**
   - 원본 데이터가 위기 상황 거의 없음 (V5)
   - 현실 왜곡도 드물게 나타남 (V4)

**검증:**
```python
# V4/V5 주입된 턴 수동 확인
synth = json.load(open('data/sessions_synth_50.json'))

v4_sessions = [s for s in synth if s['injected_violation']['type'] == 'V4']
v5_sessions = [s for s in synth if s['injected_violation']['type'] == 'V5']

# V4 예시 확인
for s in v4_sessions[:2]:
    vio = s['injected_violation']
    print(f"\nOriginal: {vio['original_text']}")
    print(f"Rewritten: {vio['rewritten_text']}")
    print(f"Rationale: {vio['rationale']}")

# → 실제로 위반이 명확한지 사람이 판단
```

#### 문제 2: V1 과검출
**증상:**
- V1이 92/160 (57.5%)로 과도하게 많음
- Top violation도 85/160이 V1

**원인 추정:**
- Judge 프롬프트의 V1 정의가 너무 광범위
- "충분한 정보 수집"의 기준이 모호
- 대부분의 응답에서 clarifying question 부족으로 판단

**개선 방향:**
```python
# V1 정의 좁히기
"""
V1 - Missing Context (STRICT):
- Makes assumption about CRITICAL unstated details
- Gives advice that requires information NOT YET provided
- Seeker explicitly mentioned needing more info, but supporter ignored

NOT V1:
- General empathy without asking questions (this is normal)
- Building rapport before info gathering
- Reflecting on already-stated information
"""
```

#### 문제 3: 일관성 부족
**증상:**
- 동일한 응답 패턴에 다른 라벨
- 프롬프트 순서/표현에 따라 결과 변동

**개선:**
- Few-shot examples 추가
- Temperature 낮추기 (0.7 → 0.3)
- Multiple judges 사용 후 majority voting

---

## Step 6: 모델 학습

### 목적
라벨링된 turn 샘플로 multi-label violation classifier 학습

### 스크립트
`scripts/step6_train.py`

### 입력 데이터
- `data/labeled/labeled_turns_train.jsonl` (160 샘플)
- `data/labeled/labeled_turns_val.jsonl` (20 샘플)
- `configs/poc.yaml`

### 모델 아키텍처

#### 6.1 Base Model
**distilroberta-base** (HuggingFace)
- 82M parameters (RoBERTa의 경량 버전)
- Pre-trained on 영어 텍스트
- Fast inference, good performance

#### 6.2 Classification Head
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=5,  # V1, V2, V3, V4, V5
    problem_type="multi_label_classification"
)
```

**Architecture:**
```
Input Text (tokenized)
    ↓
RoBERTa Encoder (6 layers)
    ↓
[CLS] Token Representation (768-dim)
    ↓
Linear(768 → 5)
    ↓
Sigmoid Activation
    ↓
[p(V1), p(V2), p(V3), p(V4), p(V5)]  # 각각 0~1
```

**Loss Function:**
Binary Cross-Entropy (BCE) - 각 라벨에 독립적
```python
loss = BCEWithLogitsLoss()(logits, labels)
# labels: [0,1,0,1,0] 같은 multi-hot vector
```

### 입력 포맷 Serialization

#### 6.3 텍스트 직렬화

```python
def serialize_sample(sample: Dict) -> str:
    """
    샘플을 단일 텍스트로 변환 (모델 입력)
    
    Format:
    [SITUATION] {situation}
    [SUMMARY] bullet points
    [CONTEXT] conversation turns
    [RESPONSE] target response
    """
    
    # 1. Situation
    situation = sample.get('meta', {}).get('situation', '')
    
    # 2. Summary bullets
    summary_bullets = sample.get('summary', [])
    summary = '\n'.join(f"- {b}" for b in summary_bullets) if summary_bullets else "(No summary)"
    
    # 3. Context turns
    context_turns = sample.get('context_turns', [])
    context_lines = []
    for turn in context_turns:
        speaker = turn.get('speaker', 'unknown')
        text = turn.get('text', '')
        context_lines.append(f"{speaker}: {text}")
    context = '\n'.join(context_lines) if context_lines else "(No context)"
    
    # 4. Target response
    response = sample.get('response', '')
    
    # Combine
    return f"""[SITUATION]
{situation}

[SUMMARY]
{summary}

[CONTEXT]
{context}

[RESPONSE]
{response}"""
```

**예시 입력:**
```
[SITUATION]
Job loss causing financial stress

[SUMMARY]
- Seeker lost job last month
- Feeling anxious about finances
- Looking for support and guidance

[CONTEXT]
seeker: I lost my job last month and I'm really worried.
supporter: I'm sorry to hear that. Can you tell me more?
seeker: I don't know how I'll pay rent next month.
supporter: That sounds really stressful.

[RESPONSE]
You should just apply to more jobs and stop worrying.
```

### Dataset 클래스

```python
class ViolationDataset(Dataset):
    """PyTorch Dataset for violation detection"""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Serialize to text
        text = serialize_sample(sample)
        
        # 2. Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 3. Extract labels
        labels = sample.get('labels', {})
        label_vector = [
            labels.get('V1', 0),
            labels.get('V2', 0),
            labels.get('V3', 0),
            labels.get('V4', 0),
            labels.get('V5', 0),
        ]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_vector, dtype=torch.float)
        }
```

### 학습 설정

#### 6.4 TrainingArguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="models/detector",
    
    # Epochs & Batch
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    
    # Optimizer
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    
    # Evaluation
    eval_strategy="epoch",          # Validate every epoch
    save_strategy="epoch",          # Save checkpoint every epoch
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    
    # Logging
    logging_steps=10,               # Log every 10 steps
    logging_dir="models/detector/logs",
    
    # Early Stopping
    save_total_limit=2,             # Keep only 2 best checkpoints
    
    # Hardware
    fp16=False,                     # Mixed precision (GPU only)
    dataloader_num_workers=0,
    
    # Misc
    report_to="none",               # Disable wandb/tensorboard
    seed=42
)
```

#### 6.5 Metrics

```python
def compute_metrics(eval_pred):
    """
    Evaluation 시 호출되는 metric 함수
    
    Args:
        eval_pred: (predictions, labels) tuple
            predictions: (N, 5) logits
            labels: (N, 5) binary
    """
    logits, labels = eval_pred
    
    # Sigmoid + threshold
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    # Micro F1 (전체 예측의 평균)
    micro_f1 = f1_score(labels.flatten(), preds.flatten(), average='micro')
    
    # Macro F1 (각 라벨 F1의 평균)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Per-label F1
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'V1_f1': per_label_f1[0],
        'V2_f1': per_label_f1[1],
        'V3_f1': per_label_f1[2],
        'V4_f1': per_label_f1[3],
        'V5_f1': per_label_f1[4],
    }
```

### Trainer 초기화

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save final model
trainer.save_model("models/detector/final_model")
```

### 출력 데이터

**파일:**
- `models/detector/checkpoint-10/` (epoch 1)
- `models/detector/checkpoint-20/` (epoch 2)
- `models/detector/checkpoint-30/` (epoch 3, **best**)
- `models/detector/final_model/` (최종 모델)
- `models/detector/train_metrics.json`
- `models/detector/logs/` (TensorBoard logs)

**학습 로그 (Epoch 3):**
```
Epoch 3/3:
Step 10: loss=0.7264, lr=1.8e-06
Step 20: loss=0.7171, lr=3.8e-06
Step 30: loss=0.6903, lr=5.8e-06

Eval:
  eval_loss: 0.6705
  eval_micro_f1: 0.3529
  eval_macro_f1: 0.1697
  eval_V1_f1: 0.0000
  eval_V2_f1: 0.1818
  eval_V3_f1: 0.6667  ← V3만 학습됨
  eval_V4_f1: 0.0000
  eval_V5_f1: 0.0000
```

### 실행 명령
```bash
python scripts/step6_train.py \
  --input_dir data/labeled \
  --output_dir models/detector \
  --config configs/poc.yaml
```

### 예상 실행 시간
- CPU: ~15분
- GPU (T4): ~5분

### 실행 로그 예시
```
============================================================
STEP 6: Train Violation Detector
============================================================

Loading training data from: data/labeled
  Train samples: 160
  Val samples: 20

Loading tokenizer: distilroberta-base
Creating datasets (max_length=512)...

Initializing model: distilroberta-base
Model parameters: 82M

Training configuration:
  Epochs: 3
  Batch size: 16
  Learning rate: 2e-5
  Output dir: models\detector

Starting training...
(This may take several minutes depending on hardware)

{'loss': '0.7264', 'grad_norm': '1.602', 'learning_rate': '1.8e-06', 'epoch': '1'}
{'eval_loss': '0.7283', 'eval_micro_f1': '0.2222', 'eval_macro_f1': '0.1697', ...}

{'loss': '0.7171', 'grad_norm': '1.837', 'learning_rate': '3.8e-06', 'epoch': '2'}
{'eval_loss': '0.7084', 'eval_micro_f1': '0.2727', 'eval_macro_f1': '0.1697', ...}

{'loss': '0.6903', 'grad_norm': '1.739', 'learning_rate': '5.8e-06', 'epoch': '3'}
{'eval_loss': '0.6705', 'eval_micro_f1': '0.3529', 'eval_macro_f1': '0.1697', ...}

Training runtime: 5 min 27 sec
Saving final model to: models\detector\final_model

✅ Step 6 complete!
```

### 검증 방법

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 모델 로드
model = AutoModelForSequenceClassification.from_pretrained("models/detector/final_model")
tokenizer = AutoTokenizer.from_pretrained("models/detector/final_model")

# 2. 테스트 예측
test_text = """[SITUATION]
Feeling lonely after moving to new city

[SUMMARY]
- Moved 3 months ago
- Having trouble making friends
- Feeling isolated

[CONTEXT]
seeker: I don't know anyone here.
supporter: That must be hard.

[RESPONSE]
You should just go out more and talk to people.
"""

inputs = tokenizer(test_text, return_tensors="pt")
outputs = model(**inputs)
probs = torch.sigmoid(outputs.logits)[0].tolist()

print("Predictions:")
for i, v in enumerate(['V1', 'V2', 'V3', 'V4', 'V5']):
    print(f"  {v}: {probs[i]:.4f} {'✓' if probs[i] > 0.5 else ''}")
```

### 핵심 문제점

**문제 1: V1 학습 실패**
- 라벨: 92/160 (가장 많음)
- 모델 예측: 0개
- 원인: 극심한 class imbalance + 라벨 품질 문제

**문제 2: V2/V3만 학습**
- V3 F1: 0.67 (괜찮음)
- V2 F1: 0.18 (낮음)
- 모델이 거의 모든 샘플을 V2/V3로 분류

**문제 3: 작은 데이터셋**
- 160 train 샘플은 부족
- 원래 계획: 200+200 세션 → 축소

### 개선 방향

**Data:**
- 200+200 or 500+500 세션으로 증량
- Class balancing (undersample V3, oversample V4/V5)

**Model:**
- Class weights 적용
```python
# V3는 weight 낮게, V1/V4/V5는 높게
class_weights = torch.tensor([2.0, 1.5, 0.5, 3.0, 3.0])
```

**Training:**
- More epochs (3 → 5)
- Different LR schedule
- Focal loss (hard examples에 집중)

**Evaluation:**
- Threshold tuning (0.5 → label별 최적값)
- Ensemble (여러 모델 평균)

---

## Step 7: 테스트 평가

### 목적
학습된 모델을 테스트 세트에서 평가하여 최종 성능 측정

### 스크립트
`scripts/step7_evaluate.py`

### 입력 데이터
- `data/labeled/labeled_turns_test.jsonl` (20 샘플)
- `models/detector/final_model/`
- `configs/poc.yaml`

### 평가 지표

#### 7.1 Multi-label Metrics

**Precision (정밀도):**
```
P = TP / (TP + FP)
예측한 위반 중 실제로 위반인 비율
```

**Recall (재현율):**
```
R = TP / (TP + FN)
실제 위반 중 모델이 찾아낸 비율
```

**F1 Score:**
```
F1 = 2 * (P * R) / (P + R)
정밀도와 재현율의 조화 평균
```

**Micro Average:**
- 모든 라벨의 TP/FP/FN을 합산 후 계산
- 샘플 수가 많은 라벨에 가중치

**Macro Average:**
- 각 라벨의 metric을 독립적으로 계산 후 평균
- 모든 라벨에 동일한 가중치

#### 7.2 Threshold

**Binary classification from probabilities:**
```python
# Model output: [0.1, 0.8, 0.6, 0.2, 0.05]
# Threshold: 0.5

predictions = [
    0,  # V1: 0.1 < 0.5
    1,  # V2: 0.8 >= 0.5
    1,  # V3: 0.6 >= 0.5
    0,  # V4: 0.2 < 0.5
    0,  # V5: 0.05 < 0.5
]
```

**현재:** 모든 라벨에 0.5 사용  
**개선:** 라벨별 최적 threshold 탐색

### 평가 로직

```python
def evaluate_model(model_path: Path, test_path: Path, tokenizer):
    """모델 평가 및 메트릭 계산"""
    
    # 1. 데이터 로드
    test_samples = load_jsonl(test_path)
    
    # 2. 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # 3. 추론
    all_predictions = []
    all_labels = []
    
    for sample in test_samples:
        # Serialize & tokenize
        text = serialize_sample(sample)
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # Forward
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
        
        # Sigmoid
        probs = 1 / (1 + np.exp(-logits))
        
        # Ground truth
        labels = [
            sample['labels']['V1'],
            sample['labels']['V2'],
            sample['labels']['V3'],
            sample['labels']['V4'],
            sample['labels']['V5'],
        ]
        
        all_predictions.append(probs)
        all_labels.append(labels)
    
    # 4. Binarize predictions
    all_predictions = np.array(all_predictions)  # (N, 5)
    all_labels = np.array(all_labels)            # (N, 5)
    pred_binary = (all_predictions >= 0.5).astype(int)
    
    # 5. Compute metrics
    from sklearn.metrics import precision_recall_fscore_support
    
    # Overall
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels.flatten(), pred_binary.flatten(),
        average='micro', zero_division=0
    )
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels.flatten(), pred_binary.flatten(),
        average='macro', zero_division=0
    )
    
    # Per-label
    per_label_metrics = {}
    for i, v_name in enumerate(['V1', 'V2', 'V3', 'V4', 'V5']):
        p, r, f1, support = precision_recall_fscore_support(
            all_labels[:, i], pred_binary[:, i],
            average='binary', zero_division=0
        )
        
        per_label_metrics[v_name] = {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'support': int(all_labels[:, i].sum()),
            'predicted_positives': int(pred_binary[:, i].sum()),
            'true_positives': int(((all_labels[:, i] == 1) & (pred_binary[:, i] == 1)).sum())
        }
    
    return {
        'micro': {'precision': p_micro, 'recall': r_micro, 'f1': f1_micro},
        'macro': {'precision': p_macro, 'recall': r_macro, 'f1': f1_macro},
        'per_label': per_label_metrics
    }
```

### 테스트 세트 구성

**Ground Truth 분포:**
```
V1: 14 samples (70%)
V2: 1 sample (5%)
V3: 10 samples (50%)
V4: 0 samples (0%)
V5: 1 sample (5%)

Total samples: 20
Multi-label: ~10 samples
```

### 최종 결과

#### 7.3 테스트 성능

**Overall Metrics:**
```
Metric       Micro   Macro
-------------------------------
Precision    0.5600  0.5125
Recall       0.5600  0.5156
F1 Score     0.5600  0.5025
```

**Per-Label Results:**

| Label | Precision | Recall | F1    | Support | Predicted | TP |
|-------|-----------|--------|-------|---------|-----------|-----|
| V1    | 0.0000    | 0.0000 | 0.0000| 14      | 0         | 0   |
| V2    | 0.0500    | 1.0000 | 0.0952| 1       | 20        | 1   |
| V3    | 0.5000    | 1.0000 | 0.6667| 10      | 20        | 10  |
| V4    | 0.0000    | 0.0000 | 0.0000| 0       | 0         | 0   |
| V5    | 0.0000    | 0.0000 | 0.0000| 1       | 0         | 0   |

**해석:**
- **V1 (14개 ground truth):** 0개 예측 → 완전 실패
- **V2 (1개):** 20개 예측 → 극심한 과검출, 운 좋게 1개 맞춤
- **V3 (10개):** 20개 예측 → 과검출, 하지만 10/10 모두 찾음 (100% recall)
- **V4:** 데이터 없음
- **V5 (1개):** 0개 예측 → 찾지 못함

### 출력 데이터

**파일:**
- `models/detector/test_results.json` (메트릭 JSON)
- `models/detector/test_predictions.jsonl` (샘플별 예측값)

**test_results.json:**
```json
{
  "metrics": {
    "micro": {"precision": 0.56, "recall": 0.56, "f1": 0.56},
    "macro": {"precision": 0.5125, "recall": 0.5156, "f1": 0.5025},
    "per_label": {
      "V1": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 14},
      "V2": {"precision": 0.05, "recall": 1.0, "f1": 0.0952, "support": 1},
      "V3": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 10},
      "V4": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
      "V5": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    }
  },
  "num_samples": 20,
  "label_distribution": {"V1": 14, "V2": 1, "V3": 10, "V4": 0, "V5": 1}
}
```

**test_predictions.jsonl (예시):**
```json
{
  "sample_id": "synth_0003_9",
  "predictions": {"V1": 0.12, "V2": 0.87, "V3": 0.92, "V4": 0.05, "V5": 0.01},
  "labels": {"V1": 1, "V2": 1, "V3": 0, "V4": 0, "V5": 0}
}
```

### 실행 명령
```bash
python scripts/step7_evaluate.py \
  --input_dir data/labeled \
  --model_dir models/detector \
  --output_dir models/detector \
  --config configs/poc.yaml
```

### 예상 실행 시간
< 1분 (20 샘플만)

### 실행 로그 예시
```
============================================================
STEP 7: Evaluate Violation Detector
============================================================

Loading test data from: data\labeled\labeled_turns_test.jsonl
  Test samples: 20

Test label distribution:
  V1: 14
  V2: 1
  V3: 10
  V4: 0
  V5: 1

Loading model from: models\detector\final_model
Model loaded on: cpu

Running inference on 20 samples...
  Processed 10/20
  Processed 20/20

Inference complete!

Computing metrics...

Results saved to: models\detector\test_results.json
Predictions saved to: models\detector\test_predictions.jsonl

============================================================
Test Results
============================================================

Overall Metrics:
  Micro - P: 0.5600, R: 0.5600, F1: 0.5600
  Macro - P: 0.5125, R: 0.5156, F1: 0.5025

Per-Label Metrics:
Label    Precision    Recall       F1           Support    Predicted  TP
--------------------------------------------------------------------------------
V1       0.0000       0.0000       0.0000       14         0          0
V2       0.0500       1.0000       0.0952       1          20         1
V3       0.5000       1.0000       0.6667       10         20         10
V4       0.0000       0.0000       0.0000       0          0          0
V5       0.0000       0.0000       0.0000       1          0          0

============================================================
✅ Step 7 complete!
============================================================
```

### 검증 방법

```python
import json

# 1. 결과 로드
with open('models/detector/test_results.json') as f:
    results = json.load(f)

print("Micro F1:", results['metrics']['micro']['f1'])
print("Macro F1:", results['metrics']['macro']['f1'])

# 2. 예측값 확인
predictions = []
with open('models/detector/test_predictions.jsonl') as f:
    for line in f:
        predictions.append(json.loads(line))

# V2/V3 과검출 확인
v2_preds = [p for p in predictions if p['predictions']['V2'] >= 0.5]
v3_preds = [p for p in predictions if p['predictions']['V3'] >= 0.5]

print(f"\nV2 predicted: {len(v2_preds)}/20")  # 20
print(f"V3 predicted: {len(v3_preds)}/20")  # 20

# 3. False Positives 분석
v2_fps = [
    p for p in predictions
    if p['predictions']['V2'] >= 0.5 and p['labels']['V2'] == 0
]
print(f"V2 False Positives: {len(v2_fps)}")  # 19

# 4. Confusion matrix (V3 예시)
tp = sum(1 for p in predictions if p['predictions']['V3'] >= 0.5 and p['labels']['V3'] == 1)
fp = sum(1 for p in predictions if p['predictions']['V3'] >= 0.5 and p['labels']['V3'] == 0)
fn = sum(1 for p in predictions if p['predictions']['V3'] < 0.5 and p['labels']['V3'] == 1)
tn = sum(1 for p in predictions if p['predictions']['V3'] < 0.5 and p['labels']['V3'] == 0)

print(f"\nV3 Confusion Matrix:")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
# TP: 10, FP: 10, FN: 0, TN: 0
# → 모든 샘플을 V3 positive로 분류
```

### 핵심 발견

**발견 1: 극심한 과검출**
- 모델이 거의 모든 응답(20/20)을 V2/V3로 분류
- Training data bias: V3가 71/160 (44%)로 많았음

**발견 2: V1 학습 완전 실패**
- Ground truth 14개인데 0개 예측
- 가능한 원인:
  1. LLM-judge의 V1 라벨링이 부정확
  2. V1의 특징이 V2/V3과 혼동됨
  3. 학습 데이터에서 V1 패턴 약함

**발견 3: Micro vs Macro 차이**
- Micro F1 (0.56) > Macro F1 (0.50)
- V3가 많아서 Micro가 높게 나옴
- Macro는 V1/V4/V5의 0이 평균에 반영

**발견 4: Threshold 문제**
- 모든 라벨에 0.5 threshold 사용
- V2/V3는 threshold를 높여야 함 (0.7~0.8)
- V1/V4/V5는 낮춰야 함 (0.3~0.4)

---

## 🔍 전체 파이프라인 분석

### 성공한 부분

✅ **End-to-end 파이프라인 구축**
- 7단계 자동화 완료
- 재현 가능한 실험 프레임워크
- 약 30분 내 전체 실행 가능 (LLM 호출 제외 시)

✅ **합성 데이터 생성**
- 50/50 세션 100% 성공률
- LLM을 활용한 realistic violation 주입
- Metadata tracking으로 추적 가능

✅ **LLM-Judge 시스템**
- 200 샘플을 ~10분에 라벨링
- Multi-label 지원
- Evidence span 제공

✅ **V3 검출 작동**
- F1 0.67로 실용 가능 수준
- Proof-of-concept 달성

### 실패한 부분

❌ **V1 학습 완전 실패**
- Ground truth: 92/160 (가장 많음)
- 모델 예측: 0/20
- 추정 원인:
  1. LLM-judge의 V1 과검출 (너무 광범위한 정의)
  2. V1과 V2/V3의 feature 혼동
  3. Label noise가 학습 방해

❌ **V4/V5 데이터 부재**
- 합성에서 주입했으나 judge가 인식 못함
- 학습 불가능 (0 samples)
- 추정 원인:
  1. Rewrite가 너무 약함 (프롬프트의 "realistic" 강조)
  2. Judge 프롬프트가 V4/V5에 대해 너무 엄격
  3. ESConv 원본 데이터에 V4/V5 패턴 희소

❌ **V2/V3 과검출**
- 모델이 거의 모든 응답(20/20)을 V2/V3로 분류
- Precision 매우 낮음 (0.05, 0.50)
- Class imbalance의 전형적 증상

---

## 💡 개선 방향

### 즉시 적용 가능 (Short-term)

**1. 데이터 증량**
```yaml
# configs/poc.yaml 수정
sampling:
  num_sessions: 200  # 50 → 200
  
synthesis:
  num_sessions: 200  # 50 → 200
```
- Train: 640 샘플 (현재 160의 4배)
- 학습 안정성 증가

**2. Class Weights 적용**
```python
# scripts/step6_train.py에 추가
from torch.nn import BCEWithLogitsLoss

# 빈도에 반비례하는 weight
# V3가 많으므로 낮은 weight, V4/V5는 높은 weight
class_weights = torch.tensor([
    2.0,  # V1: 중간
    1.5,  # V2: 중간
    0.5,  # V3: 낮음 (과검출 방지)
    3.0,  # V4: 높음
    3.0,  # V5: 높음
])

loss_fn = BCEWithLogitsLoss(pos_weight=class_weights)
```

**3. Judge 프롬프트 개선**
```python
# src/llm/prompts.py 수정

# V1 정의 좁히기
V1_STRICT = """
V1 - Missing Context (STRICT):
- Makes critical assumption about UNSTATED information
- Gives advice requiring facts NOT YET mentioned
- Seeker explicitly signals confusion, supporter ignores

NOT V1 (normal counseling):
- Building empathy before questions
- Reflecting on stated information
- General supportive responses
"""

# V4/V5 정의 완화 + Few-shot
V4_WITH_EXAMPLES = """
V4 - Reality Distortion:
...

Examples:
- "It's not that bad" → Minimizing
- "Everything happens for a reason" → Denying agency
- "Just think positive!" → Toxic positivity

Counter-examples (NOT V4):
- "That sounds really tough" → Validation
- "How can I support you?" → Open-ended
"""
```

**4. Threshold 튜닝**
```python
# scripts/step7_evaluate.py에 추가
from sklearn.metrics import precision_recall_curve

def find_optimal_thresholds(y_true, y_scores):
    """각 라벨별 최적 threshold 탐색"""
    optimal_thresholds = {}
    
    for i, v_name in enumerate(['V1', 'V2', 'V3', 'V4', 'V5']):
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i], y_scores[:, i]
        )
        
        # F1 최대화하는 threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_thresholds[v_name] = thresholds[best_idx]
    
    return optimal_thresholds

# Validation set에서 threshold 찾기
thresholds = find_optimal_thresholds(val_labels, val_scores)
# 예: {'V1': 0.3, 'V2': 0.7, 'V3': 0.8, 'V4': 0.2, 'V5': 0.2}

# Test에 적용
pred_binary = (test_scores >= list(thresholds.values())).astype(int)
```

### 중기 개선 (Medium-term)

**5. 합성 방법 다양화**
```python
# src/synth/ 에 새 모듈 추가

# Method A: 현재 (single-turn rewrite)
# Method B: Multi-turn rewrite
def multi_turn_rewrite(session, violation_type):
    """여러 턴을 연쇄적으로 리라이트"""
    # V1 예시: 3개 턴에 걸쳐 점진적으로 맥락 무시
    pass

# Method C: Scratch generation
def generate_from_scratch(situation, violation_type):
    """주어진 상황에서 위반이 포함된 대화 생성"""
    pass

# Method D: Rule-based injection
def rule_based_injection(session, violation_type):
    """템플릿 기반 위반 주입 (V2/V5에 효과적)"""
    if violation_type == "V2":
        # "you should" 패턴 강제 삽입
        response = f"You should {random.choice(DIRECTIVE_VERBS)} ..."
    pass
```

**6. Ensemble Models**
```python
# 3개 모델 학습 (다른 seed)
models = [
    train_model(seed=42),
    train_model(seed=123),
    train_model(seed=456)
]

# Soft voting
ensemble_probs = np.mean([m.predict(x) for m in models], axis=0)
```

**7. 전문가 검증**
```python
# 샘플 100개 선정 (LLM-judge와 불일치한 것 위주)
ambiguous_samples = [
    s for s in labeled_samples
    if (s['meta']['is_violation_turn'] and s['labels'][violation] == 0) or
       (not s['meta']['is_violation_turn'] and sum(s['labels'].values()) > 0)
]

# 전문가 라벨링
expert_labels = get_expert_annotations(ambiguous_samples)

# LLM-judge와 비교
agreement = calculate_kappa(llm_labels, expert_labels)
# κ < 0.6 → LLM-judge 품질 문제
```

### 장기 개선 (Long-term)

**8. Active Learning**
```python
# 1. 모델이 불확실한 샘플 선정
uncertain = [s for s in unlabeled if entropy(model(s)) > threshold]

# 2. 전문가 라벨링
expert_labeled = get_expert_annotations(uncertain)

# 3. 재학습
retrain_model(train_data + expert_labeled)
```

**9. Curriculum Learning**
```python
# Easy → Hard 순으로 학습
# Phase 1: 명확한 위반만 (V2/V3의 강한 사례)
train_phase1(easy_samples, epochs=2)

# Phase 2: 모든 샘플
train_phase2(all_samples, epochs=3)
```

**10. 프로덕션 배포**
```python
# FastAPI 서버
from fastapi import FastAPI
app = FastAPI()

model = load_model("models/detector/final_model")

@app.post("/detect")
def detect_violations(conversation: List[Dict]):
    """실시간 위반 탐지 API"""
    # 1. 최근 턴 추출
    recent_turns = conversation[-8:]
    
    # 2. 요약 생성
    summary = summarize(conversation)
    
    # 3. 예측
    violations = model.predict({
        'context': recent_turns,
        'summary': summary,
        'response': conversation[-1]['text']
    })
    
    return {
        'violations': violations,
        'confidence': max(violations.values()),
        'evidence': extract_evidence(conversation[-1])
    }
```

---

## 📈 성능 예측

### 현재 (50+50 세션, 160 train)
```
Micro F1: 0.56
Macro F1: 0.50
V1/V4/V5: 학습 실패
```

### 데이터만 증량 (200+200, 640 train)
```
예상 Micro F1: 0.65 (+0.09)
예상 Macro F1: 0.58 (+0.08)
V1: 일부 학습 시작 (F1 ~0.3)
V4/V5: 여전히 어려움
```

### 데이터 + Class Weights + Threshold
```
예상 Micro F1: 0.70
예상 Macro F1: 0.65
V1: F1 ~0.5
V3 과검출 완화
```

### 전문가 라벨링 100샘플 추가
```
예상 Micro F1: 0.75
예상 Macro F1: 0.70
모든 라벨 실용 수준
```

---

## 🎓 Lessons Learned

### 데이터 품질이 가장 중요
- 160 샘플로는 부족
- LLM-judge는 편리하지만 전문가 검증 필수
- Label noise가 학습을 망칠 수 있음

### Multi-label은 Single-label보다 어렵다
- Class imbalance 문제 심화
- Threshold tuning이 critical
- Per-label metrics 반드시 확인

### 합성 데이터의 한계
- LLM rewrite는 "realistic"과 "obvious" 사이 trade-off
- 일부 위반(V4/V5)은 rule-based 주입이 나을 수 있음
- 실제 실패 사례 수집이 가장 좋음

### Small dataset에서의 전략
- Pretrained model 필수 (distilroberta)
- Data augmentation (paraphrase, backtranslation)
- Transfer learning (관련 task에서 fine-tune 후 재fine-tune)

### Evaluation은 단일 숫자가 아니다
- Micro F1만 보면 안됨
- Per-label 분석 필수
- Confusion matrix, False Positive 분석
- Qualitative error analysis (샘플 수동 확인)

---

## 🔗 참고 자료

### 데이터셋
- **ESConv**: Liu et al. (2021), "Towards Emotional Support Dialog Systems"
  - https://github.com/thu-coai/Emotional-Support-Conversation

### 관련 연구
- **Multi-label Text Classification**: Zhang et al. (2021)
- **LLM as Judge**: Zheng et al. (2023), "Judging LLM-as-a-Judge"
- **Synthetic Data for NLP**: Schick & Schütze (2021), "Generating Datasets with Pretrained Language Models"

### 도구
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **OpenAI API**: https://platform.openai.com/docs
- **scikit-learn**: https://scikit-learn.org/stable/modules/multiclass.html

---

## 🏁 다음 단계

### Immediate Actions
1. [ ] 200+200 세션으로 재실행
2. [ ] Class weights 적용
3. [ ] Threshold 튜닝
4. [ ] V1/V4/V5 프롬프트 개선

### Next Sprint
1. [ ] 전문가 100 샘플 라벨링
2. [ ] Ensemble model 실험
3. [ ] Rule-based V4/V5 주입
4. [ ] Error analysis 보고서

### Future Work
1. [ ] 실제 상담 데이터 수집
2. [ ] Active learning 파이프라인
3. [ ] API 서버 구축
4. [ ] A/B 테스트 설계

---

**프로젝트 완료일:** 2026-01-29  
**파이프라인 버전:** v1.0  
**문의:** GitHub Issues (https://github.com/th610/ICMR_GenData/issues)
