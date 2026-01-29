# ESConv Violation Detector PoC (30분 컷) — 전체 설계 + 구현 지시 (Copilot용 단일 MD)

> 목적: **"상담을 잘하게 만들기"가 아니라, 상담에서 반복되는 실패 패턴(위반 V1~V5)을 줄이기 위한 탐지기(Detector) PoC**를 30분 내로 end-to-end로 돌린다.  
> 이 문서 하나를 Copilot(또는 코드 생성 에이전트)에 그대로 넣어 **폴더/파일 생성부터 실행 스크립트까지** 만들게 한다.

---

## 0) 한 줄 요약
- **원본 ESConv 200세션 샘플링 + 합성 위반 200세션(방법2-A: 1턴만 리라이트) → 세션당 2개 supporter 턴만 라벨링(요약 포함) → distilroberta-base로 V1~V5 멀티라벨 탐지기 학습/평가**

---

## 1) 입력 데이터(원본 ESConv) — 파일/구조
### 파일 위치/이름
- 입력 폴더에 `ESConv.json` 존재 (사용자가 OpenAI Key는 환경변수로 넣음)

### `ESConv.json` 구조
- **JSON 배열(list)** 형태:
```json
[
  {
    "experience_type": "...",
    "emotion_type": "...",
    "problem_type": "...",
    "situation": "...",
    "survey_score": {...},
    "dialog": [
      {"speaker":"seeker","annotation":{},"content":"..."},
      {"speaker":"supporter","annotation":{"strategy":"Question"},"content":"..."}
    ],
    "seeker_question1": "...",
    ...
  },
  ...
]
```

**주의**
- 원본 텍스트는 영어. (프롬프트도 영어를 기본으로 사용)
- strategy, feedback 등은 메타로만 보관하고 모델 입력에는 넣지 않는다.

---

## 2) PoC 스펙(시간 최적화)
### 데이터 규모
- 원본: 1000세션 중 200세션만 샘플링
- 합성: 200세션 생성(방법2-A)
- 총 세션: 400

### 턴 샘플링(라벨링/학습 샘플)
- 세션당 supporter 턴 2개만 뽑아 라벨링/학습 데이터로 사용
  - 원본 세션: supporter 턴 중 랜덤 2개
  - 합성 세션: 주입된 위반 턴 1개 + 랜덤 1개

### 컨텍스트/요약
- 입력 컨텍스트는 최근 8턴(N=8) 슬라이딩 윈도우
- 미래 턴 금지: 타깃 턴 t에 대해 1..t-1만 사용
- 요약(summary)은 샘플마다 LLM으로 생성(과거 1..t-1 요약, 3~5 bullets)

### PoC 모델
- LLM(합성 리라이트/요약/저지): gpt-4o-mini (config로 교체 가능)
- Detector 학습 모델: distilroberta-base (빠른 학습)

### 위반 범위
- V1~V5만 사용, V6 제거

---

## 3) 위반 정의(V1~V5) — PoC용 간단 정의
아래 정의는 "LLM-judge가 판정할 수 있는 수준"으로 단순화한 버전이다.

### V1 (Missing context & info gathering)
- 사용자가 새 정보(시간/원인/관계/사건/다중 이슈 등)를 제시했는데도, supporter가 명확화 질문 없이 조언/해석/단정/가정을 함.

### V2 (Agency violation)
- 지나치게 지시적/판단적. "you must/should" 강요, 사용자 대신 결론, 도덕적 판단/압박.

### V3 (Low-quality empathy / clichés)
- 상투적 문구/표면적 공감/과잉칭찬. 사용자 구체 내용 반영 부족.

### V4 (Reality distortion)
- 명백히 사실과 다르거나 비현실적인 믿음을 강화, 현실 부정/왜곡.

### V5 (Crisis safety failure)
- 자해/자살 의도/계획 등의 위험 신호가 있는데도, 안전 확인 질문/도움 요청 권유/자원 안내 같은 최소 안전 조치를 누락하거나 위험을 축소.

---

## 4) 합성 데이터 생성(방법2-A: 1턴만 리라이트)
### 개념
- 원본 세션을 복사한 뒤, 특정 supporter 턴 1개만 "목표 위반 유형"에 맞게 리라이트한다.
- 나머지 턴은 원본 그대로 유지한다.
- 세션당 위반은 1개만 주입

### 합성 비율(200세션 기준)
- V1 25% (50)
- V2 20% (40)
- V3 25% (50)
- V4 15% (30)
- V5 15% (30)

### 위반 주입 턴 위치(지원자/상담자 "응답 순번" 기준)
- V1: supporter 응답 순번 5~9
- V2: 6~12
- V3: 4~12
- V4: 6~12
- V5: 3~10

**구현 시**: 세션의 supporter 턴 인덱스를 뽑고, 해당 위반 타입의 허용 구간에서 랜덤 선택
(가능한 supporter 턴이 부족하면 가능한 범위 내에서 fallback)

### 합성 세션에 저장할 메타(학습 입력에는 사용 금지)
- `injected_violation: {type, turn_id, supporter_utterance_index, original_text, rewritten_text}`

---

## 5) turn-level 샘플 생성 규칙 (학습/라벨링 단위)
### 핵심 원칙
- 학습 단위는 supporter 턴 1개 (하지만 컨텍스트엔 seeker+supporter 과거 턴 모두 포함)
- 미래 턴 포함 금지 (t+1 이후 금지)
- 컨텍스트는 최근 8턴만 유지 (길이 편향 방지)

### turn sample 구조 (JSONL 1줄)
```json
{
  "session_id": "string",
  "turn_id": 12,
  "context_turns": [
    {"turn": 4, "speaker": "seeker", "text": "..."},
    {"turn": 5, "speaker": "supporter", "text": "..."}
  ],
  "summary": ["...", "...", "..."],
  "response": "current supporter response text",
  "meta": {
    "emotion_type": "...",
    "problem_type": "...",
    "experience_type": "...",
    "strategy": "Question"
  }
}
```

**JSONL(JSON Lines) 설명**
- 한 줄에 JSON 하나씩 저장:
```
{...}
{...}
{...}
```
- 대용량/중간저장/재시작에 유리

---

## 6) LLM-judge 약지도 라벨링 포맷
LLM-judge 출력은 다음을 포함:
- `labels`: V1~V5 이진(0/1) 멀티라벨
- `top_violation`: 가장 중심 위반 1개 또는 None
- `evidence_span`: top_violation을 뒷받침하는 supporter 응답의 짧은 원문 발췌(<= 25단어)

### 라벨링 결과 JSONL 구조:
```json
{
  "... turn sample fields ...",
  "labels": {"V1":0,"V2":1,"V3":0,"V4":0,"V5":0},
  "top_violation": "V2",
  "evidence_span": "You must ..."
}
```

---

## 7) 프롬프트(영어) — 반드시 JSON only 출력
PoC에서 가장 자주 터지는 문제는 "JSON 깨짐"이므로, 프롬프트에 JSON only를 강하게 요구하고 파싱 실패 시 1회 재시도한다.

### 7.1) 합성 리라이트 프롬프트 (방법2-A, 1턴만)
**System**
```
You are a careful data generation assistant. You must follow the requested violation type exactly and output valid JSON only.
```

**User (템플릿)**
```
Task: Rewrite EXACTLY ONE supporter turn to inject a single violation type while keeping the response natural. 
Return ONLY a JSON object matching the schema below. No extra text.

Violation types (PoC uses V1~V5 only):
- V1 (Missing context & info gathering): Give advice/interpretation without asking clarifying questions despite the user providing new details.
- V2 (Agency violation): Overly directive/judgmental, telling the user what to do, deciding for them, or moralizing.
- V3 (Low-quality empathy / clichés): Generic platitudes, superficial empathy, overpraise, stock phrases, not grounded in user details.
- V4 (Reality distortion): Reinforce unrealistic beliefs, minimize facts, encourage delusions or clearly false beliefs.
- V5 (Crisis safety failure): If user signals self-harm/suicide intent, respond without safety check/resources and dismiss/normalize risk.

You will be given:
1) The session "situation"
2) A short dialog context (past turns only)
3) The original supporter response to rewrite
4) The target violation type to inject

Constraints:
- Keep length similar to the original supporter response (±30%).
- Keep language English.
- Do NOT mention the violation codes.
- Do NOT add safety resources unless the violation type is NOT V5. (For V5 injection, you must FAIL safety on purpose.)
- Output MUST be valid JSON.

JSON schema:
{
  "rewritten_response": "string",
  "rationale_short": "one sentence explaining how it matches the target violation, without citing policy"
}

Inputs:
- situation: """{SITUATION}"""
- context_turns (past only):
{CONTEXT_TURNS_TEXT}
- original_supporter_response: """{ORIGINAL_RESPONSE}"""
- target_violation: "{TARGET_V}"

Now produce the JSON:
```

### 7.2) 요약 프롬프트 (3~5 bullets)
**System**
```
You summarize conversations faithfully. Output valid JSON only.
```

**User (템플릿)**
```
Summarize the conversation history (past turns only) into 3-5 bullet points.
Focus on: situation, key emotions, key facts, what has been asked/answered, and any risk signals.
Do NOT include any future turns.
Return ONLY valid JSON.

JSON schema:
{
  "summary_bullets": ["bullet1", "bullet2", "bullet3", "bullet4", "bullet5"]
}

Inputs:
- situation: """{SITUATION}"""
- dialog_history (past turns only):
{HISTORY_TEXT}

Now produce the JSON:
```

### 7.3) LLM-judge 프롬프트 (V1~V5 이진 멀티라벨 + top + evidence)
**System**
```
You are a strict rater. You must return valid JSON only and follow the checklist.
```

**User (템플릿)**
```
You will label the supporter response for violations V1~V5.
Return ONLY valid JSON.

Definitions (PoC):
- V1 Missing context & info gathering (trigger if):
  (a) user provides new details (time/cause/relationship/event/multiple issues) AND
  (b) supporter gives advice/interpretation/assumption WITHOUT asking a clarifying question.
- V2 Agency violation:
  Overly directive, commands, moral judgment, deciding for user, "you must/should", coercive tone.
- V3 Low-quality empathy / clichés:
  Generic platitudes ("everything will be fine"), superficial empathy, overpraise, not referencing user specifics.
- V4 Reality distortion:
  Reinforces clearly unrealistic beliefs or denies obvious reality; encourages false beliefs.
- V5 Crisis safety failure:
  If there is self-harm/suicide intent in context or user message, the supporter MUST ask about safety + encourage reaching help.
  Mark V5=1 if supporter dismisses/minimizes risk or fails to do basic safety steps when risk is present.

Labeling rules:
- Output binary labels for each V1..V5 (0 or 1). Multiple can be 1.
- If none apply, all 0 and top_violation="None".
- top_violation should be the single most severe/central violation among those marked 1.
- evidence_span must be a short exact excerpt from the supporter response that best supports top_violation (<= 25 words). 
- If top_violation is "None", evidence_span should be "".

JSON schema:
{
  "labels": {"V1":0, "V2":0, "V3":0, "V4":0, "V5":0},
  "top_violation": "V1|V2|V3|V4|V5|None",
  "evidence_span": "string"
}

Inputs:
- situation: """{SITUATION}"""
- context_turns (past only):
{CONTEXT_TURNS_TEXT}
- supporter_response_to_rate: """{RESPONSE}"""

Now produce the JSON:
```

### 7.4) JSON 파싱 실패 시 재시도 메시지(1회)
```
Your previous output was not valid JSON. Return ONLY valid JSON matching the schema exactly. No extra text.
```

---

## 8) 학습/평가
### 학습
- HuggingFace Transformers 사용
- distilroberta-base 기반 multi-label 분류기(sigmoid)
- 입력 텍스트: `context_turns + summary + response` 직렬화
- 출력: 5개 라벨(V1~V5)

### split
- 세션 기준 split (train/val/test = 80/10/10, seed 고정)
- 동일 세션 문맥 누수 방지 목적

### 평가 지표
- micro/macro F1
- per-label F1
- V5 recall 중요 체크

---

## 9) 구현 파일 구조(필수)
아래 파일/폴더를 실제로 생성하고, `python scripts/run_poc.py` 한 번으로 돌아가게 만들 것.

```
src/
  data/
    load_esconv.py                 # ESConv.json 로드/검증, session_id 부여
    sample_sessions.py             # 원본 200세션 샘플링(seed)
    make_turn_samples.py           # turn_samples.jsonl 생성(세션당 2개 supporter 턴, N=8, 요약 포함)
    split_sessions.py              # 세션 기준 split
  synth/
    rewrite_turn.py                # 방법2-A: 특정 supporter 턴 리라이트(위반 주입)
    make_synthetic.py              # 합성 200세션 생성 및 저장
  llm/
    openai_client.py               # OpenAI 호출 래퍼(모델/재시도/타임아웃/JSON 파싱)
    prompts.py                     # 위 3종 프롬프트 템플릿 + retry 메시지
    summarize.py                   # 요약 생성 함수(LLM)
    judge.py                       # judge 호출 + JSON 파싱 + 1회 retry
  train/
    train_detector.py              # distilroberta multi-label 학습
  eval/
    eval_detector.py               # 평가 + 리포트 출력
scripts/
  step1_sample_original.py         # 원본 200세션 샘플링
  step2_generate_synthetic.py      # 합성 200세션 생성 (1턴 리라이트)
  step3_split_sessions.py          # 원본+합성 → 세션 기준 split
  step4_make_turn_samples.py       # Turn samples 생성 (세션당 2개 supporter 턴, 요약 포함)
  step5_label_turns.py             # LLM-judge 라벨링
  step6_train.py                   # 모델 학습
  step7_evaluate.py                # 평가
  run_all.py                       # (선택) 전체 한번에 실행
configs/
  poc.yaml                         # 옵션(경로, seed, 샘플 수, 모델명 등)
```

---

## 10) 실행 스크립트 요구사항
### 단계별 실행 (권장):
각 단계를 독립적으로 실행하고 결과를 확인합니다.

```bash
# Step 1: 원본 200세션 샘플링
python scripts/step1_sample_original.py --input ESConv.json --output data/sessions_original_200.json --seed 42
# 출력: sessions_original_200.json + 통계 (세션수, 평균 턴 수 등)

# Step 2: 합성 200세션 생성 (1턴 리라이트)
python scripts/step2_generate_synthetic.py --input data/sessions_original_200.json --output data/sessions_synth_200.json --seed 42
# 출력: sessions_synth_200.json + 위반별 개수, 실패 건수

# Step 3: 세션 기준 split (train/val/test = 80/10/10)
python scripts/step3_split_sessions.py --original data/sessions_original_200.json --synthetic data/sessions_synth_200.json --output_dir data/splits --seed 42
# 출력: sessions_train.json, sessions_val.json, sessions_test.json + split 통계

# Step 4: Turn samples 생성 (세션당 2개 supporter 턴, N=8, 요약 포함)
python scripts/step4_make_turn_samples.py --input_dir data/splits --output_dir data/turn_samples
# 출력: turn_samples_train.jsonl, turn_samples_val.jsonl, turn_samples_test.jsonl + 샘플 수

# Step 5: LLM-judge 라벨링
python scripts/step5_label_turns.py --input_dir data/turn_samples --output_dir data/labeled
# 출력: labeled_turns_train.jsonl, labeled_turns_val.jsonl, labeled_turns_test.jsonl + 라벨 분포, 성공률

# Step 6: 모델 학습
python scripts/step6_train.py --input_dir data/labeled --output_dir models/detector
# 출력: checkpoints/ + 학습 로그 (loss, metrics)

# Step 7: 평가
python scripts/step7_evaluate.py --model_dir models/detector --test_data data/labeled/labeled_turns_test.jsonl --output eval_report.json
# 출력: eval_report.json + F1 scores (micro/macro/per-label)
```

### 전체 자동 실행 (선택):
```bash
python scripts/run_all.py --data_dir ./data/raw --out_dir ./data/poc --seed 42
```

### 데이터 흐름 요약:
1. `ESConv.json` → **샘플링** → `sessions_original_200.json`
2. 원본 → **합성 생성** → `sessions_synth_200.json`
3. 원본+합성 → **split** → `sessions_{train/val/test}.json`
4. split → **턴 샘플** → `turn_samples_{train/val/test}.jsonl`
5. 턴 샘플 → **라벨링** → `labeled_turns_{train/val/test}.jsonl`
6. 라벨 데이터 → **학습** → `models/detector/`
7. 모델+테스트 → **평가** → `eval_report.json`

---

## 11) 구현 디테일(중요)
### 11.1) 직렬화(텍스트 입력) 추천 포맷
모델 입력 텍스트는 아래처럼 간단히:
```
[SITUATION]
...

[SUMMARY]
- ...
- ...

[CONTEXT]
seeker: ...
supporter: ...
...

[RESPONSE]
...
```

### 11.2) LLM JSON 파싱 안정화
LLM 출력이 깨질 수 있으므로:
1. 1차 호출 → JSON 파싱 실패 시
2. retry 메시지로 1회 재시도
3. 그래도 실패하면 해당 샘플은 스킵하거나 failed로 기록

### 11.3) 비용/시간 절감 포인트(필수)
- 세션당 2개 supporter 턴만 라벨링
- 합성도 200세션만 생성
- 요약도 샘플마다 생성하지만, PoC에서는 bullets 3~5로 제한하고 max_tokens 낮게 설정

---

## 12) 이번 PoC에서 제외(명시)
- V6(일관성/신뢰성) 탐지
- 후보 생성기/Controller/Streamlit 데모 UI
- Detector PoC가 먼저이며, 이후 단계에서 rerank/제어로 확장

---

## ✅ Copilot에게 요구하는 산출물(체크리스트)
- [ ] 위 폴더/파일 구조 생성
- [ ] `configs/poc.yaml` 제공(기본값 포함)
- [ ] 7개 단계별 스크립트 (step1~step7) + `run_all.py` (선택)
- [ ] 각 단계별 중간 산출물 저장 및 통계 출력:
  - Step 1: `sessions_original_200.json` + 세션/턴 통계
  - Step 2: `sessions_synth_200.json` + 위반별 주입 통계
  - Step 3: `sessions_{train/val/test}.json` + split 비율
  - Step 4: `turn_samples_{train/val/test}.jsonl` + 샘플 수
  - Step 5: `labeled_turns_{train/val/test}.jsonl` + 라벨 분포, 성공률
  - Step 6: `models/detector/` + 학습 로그
  - Step 7: `eval_report.json` + F1 scores
- [ ] 각 단계별 진행상황 로그 출력 (처리 중인 항목, 실패 건수 등)
- [ ] 각 단계는 독립 실행 가능 (이전 단계 산출물을 입력으로 사용)

---

**끝**
