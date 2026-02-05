# 파일럿 단계별 실행 가이드
## 기존 코드 활용 + 점진적 수정 전략

---

## 🎯 Phase 0: 준비 (5분)

### 작업
- [x] 불필요한 파일 삭제 완료
- [x] `scripts/pilot/`, `data/pilot/` 폴더 생성 완료

### 다음 단계로 가는 조건
✅ 폴더 구조 확인

---

## 📝 Phase 1: 설정 파일 작성 (10분)

### 작업
- [ ] `configs/pilot.yaml` 작성
- [ ] 기존 `configs/poc.yaml` 참고

### 검증 방법
```bash
python -c "from src.utils import load_yaml; c=load_yaml('configs/pilot.yaml'); print(c)"
```

### 통과 조건
- ✅ YAML 파싱 성공
- ✅ 필수 키 존재 (normal, v1_v3, v4_v5, llm, etc)

### 예상 이슈
- YAML 문법 오류

---

## 🎨 Phase 2: 새 프롬프트 작성 (30분)

### 작업
- [ ] `src/llm/prompts_v2.py` 작성
  - [ ] V1~V3 재작성 프롬프트
  - [ ] V4/V5 멀티턴 생성 프롬프트
  - [ ] Judge 프롬프트 (선택)

### 검증 방법
```bash
python -c "from src.llm.prompts_v2 import *; print('V1 system:', V1_SYSTEM[:50]); print('V4 user:', V4_USER_TEMPLATE[:50])"
```

### 통과 조건
- ✅ 프롬프트 import 성공
- ✅ 모든 프롬프트 템플릿 존재

### 예상 이슈
- None

---

## 📦 Phase 3: Normal 세션 준비 (20분)

### 작업
- [ ] `src/generation/normal_cutter.py` 작성
- [ ] `scripts/pilot/step1_prepare_normal.py` 작성

### 핵심 로직
```python
def cut_session(session, target_length=(15, 20)):
    """ESConv 세션을 15~20턴으로 자르기"""
    dialog = session['dialog']
    if len(dialog) > max(target_length):
        cut_point = random.randint(min(target_length), max(target_length))
        session['dialog'] = dialog[:cut_point]
    return session
```

### 실행
```bash
python scripts/pilot/step1_prepare_normal.py
```

### 검증 방법
```bash
python -c "import json; s=json.load(open('data/pilot/normal/normal_0.json')); print(f'Turns: {len(s[\"dialog\"])}')"
```

### 통과 조건
- ✅ `data/pilot/normal/` 에 5개 파일 생성
- ✅ 각 세션 15~20턴
- ✅ primary_label = "Normal"
- ✅ 수동 확인: 대화가 자연스럽게 끝나는가?

### 예상 이슈
- 너무 짧게 잘리면 맥락 부족
- 해결: cut_point 조정

---

## 🔄 Phase 4: V1~V3 세션 생성 (1시간)

### 작업
- [ ] `src/generation/v1_v3_rewriter.py` 작성
  - 기존 `src/synth/rewrite_turn.py` 참고
  - 마지막 턴만 재작성
- [ ] `scripts/pilot/step2_generate_v1_v3.py` 작성

### 핵심 수정
```python
# 기존: 임의의 턴 선택
# 신규: 마지막 Supporter 턴만

def get_last_supporter_turn(dialog):
    for i in range(len(dialog)-1, -1, -1):
        if dialog[i]['speaker'] == 'supporter':
            return i
    return None
```

### 실행
```bash
python scripts/pilot/step2_generate_v1_v3.py
```

### 검증 방법
```bash
# 각 클래스 1개씩 확인
python scripts/pilot/check_samples.py --classes V1,V2,V3
```

### 통과 조건
- ✅ 각 클래스 5개씩 생성
- ✅ violation_turn_index = 마지막 supporter 턴
- ✅ violation_reason 존재
- ✅ 수동 확인:
  - V1: 정보 수집 없이 조언
  - V2: 명령조/지시적
  - V3: 표면적 공감만

### 예상 이슈
- LLM이 다른 위반을 섞음
- 해결: 프롬프트 강화 "오직 V1만", "V2/V3 하지 마세요"

---

## ⚡ Phase 5: V4/V5 세션 생성 (1시간)

### 작업
- [ ] `src/generation/v4_v5_generator.py` 작성 (완전 신규)
- [ ] `scripts/pilot/step3_generate_v4_v5.py` 작성

### 핵심 로직
```python
def generate_v4_session(llm_client, length=(10, 15)):
    """V4 멀티턴 대화 생성"""
    prompt = build_v4_prompt(target_length=length)
    result = llm_client.call(V4_SYSTEM, prompt)
    return result  # {primary_label, dialog, violation_turn_index, ...}
```

### 실행
```bash
python scripts/pilot/step3_generate_v4_v5.py
```

### 검증 방법
```bash
python scripts/pilot/check_samples.py --classes V4,V5
```

### 통과 조건
- ✅ 각 5개씩 생성
- ✅ 세션 길이 10~15턴
- ✅ 수동 확인:
  - V4: Seeker의 비현실적 신념 → Supporter가 강화
  - V5: 자해/자살 언급 → Supporter가 안전 절차 무시

### 예상 이슈
- LLM이 위기 맥락을 자연스럽게 만들지 못함
- 해결: few-shot 예시 추가

---

## ✅ Phase 6: 품질 수동 검증 (1시간)

### 작업
- [ ] `scripts/pilot/step4_manual_review.py` 작성
  - 각 클래스별로 샘플 출력
  - 사람이 검토

### 실행
```bash
python scripts/pilot/step4_manual_review.py
```

### 출력 예시
```
=== V1 샘플 1 ===
Session: v1_0.json
Violation turn: 18

[Dialog]
...

[Original last turn]
...

[Rewritten last turn]
"You should definitely apply for that job."

[Reason]
조언을 했으나 사용자의 상황을 명확히 파악하는 질문이 없었음.

✅ 적절 / ❌ 부적절 / ⚠️ 애매함
>>> 
```

### 통과 조건
- ✅ 각 클래스 최소 4/5 적절
- ⚠️ 애매한 샘플은 재생성

### 예상 이슈
- V4/V5 품질 낮음
- 해결: 프롬프트 수정 후 재생성

---

## 🔀 Phase 7: Session-level Split (10분)

### 작업
- [ ] `scripts/pilot/step5_split_sessions.py` 작성
- 기존 `scripts/step3_split_sessions.py` 수정

### 실행
```bash
python scripts/pilot/step5_split_sessions.py
```

### 검증 방법
```bash
ls data/pilot/processed/
# train.json (24 sessions)
# val.json (3 sessions)
# test.json (3 sessions)
```

### 통과 조건
- ✅ Train: 24, Val: 3, Test: 3
- ✅ 각 split에 모든 클래스 최소 1개씩
- ✅ 같은 세션이 여러 split에 안 들어감

### 예상 이슈
- Stratified split이 안될 수 있음 (30개라 작음)
- 해결: 수동 조정 OK

---

## 🎯 Phase 8: 학습 입력 생성 (1시간)

### 작업
- [ ] `src/preprocessing/dynamic_summary.py` 작성
- [ ] `src/preprocessing/input_builder.py` 작성
- [ ] `scripts/pilot/step6_create_inputs.py` 작성

### 핵심 로직
```python
def create_dynamic_summary(dialog, target_turn_idx, llm_client):
    """타겟 턴 이전까지만 요약"""
    past_turns = dialog[:target_turn_idx]
    summary = llm_client.call(SUMMARY_SYSTEM, build_summary_prompt(past_turns))
    return summary  # max 6 bullets, 150 tokens

def build_model_input(session):
    """[Summary] + [Last 4 turns] + [Target]"""
    target_idx = session['violation_turn_index']
    
    summary = create_dynamic_summary(session['dialog'], target_idx)
    last_4 = get_last_n_turns(session['dialog'], target_idx, n=4)
    target = session['dialog'][target_idx]['content']
    
    return {
        'text': format_input(summary, last_4, target),
        'label': session['primary_label']
    }
```

### 실행
```bash
python scripts/pilot/step6_create_inputs.py
```

### 검증 방법
```bash
python -c "import json; s=json.loads(open('data/pilot/processed/train.jsonl').readline()); print(len(s['text'].split())); print(s['label'])"
```

### 통과 조건
- ✅ train.jsonl (24), val.jsonl (3), test.jsonl (3) 생성
- ✅ 각 샘플 512 tokens 이내
- ✅ 데이터 누수 없음 (summary가 target 이전만 포함)
- ✅ 수동 확인: 입력 형식이 자연스러운가?

### 예상 이슈
- 토큰 길이 초과
- 해결: summary bullet 수 줄이기 (6→4)

---

## 🤖 Phase 9: 모델 학습 (30분)

### 작업
- [ ] `src/training/session_classifier.py` 작성
- [ ] `scripts/pilot/step8_train.py` 작성
- 기존 `scripts/step6_train.py` 참고

### 핵심 수정
```python
# 기존: Multi-label (BCE Loss)
# 신규: Single-label (CrossEntropy)

num_labels = 6  # Normal, V1, V2, V3, V4, V5
loss_fn = nn.CrossEntropyLoss()

# Class weight (V4/V5 boost)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
```

### 실행
```bash
python scripts/pilot/step8_train.py
```

### 검증 방법
```bash
# 학습 로그 확인
# Val loss 감소하는가?
# Macro F1 > 0.3?
```

### 통과 조건
- ✅ 학습 완료 (에러 없음)
- ✅ Val Macro F1 > 0.3 (작은 데이터라 낮아도 OK)
- ✅ Confusion matrix에서 각 클래스 최소 1개 맞춤
- ✅ Normal vs Violation 구분은 잘 됨

### 예상 이슈
- 과적합 (데이터 작음)
- 해결: Epoch 줄이기, Dropout 추가

---

## 🎉 Phase 10: 결과 분석 및 결정 (30분)

### 작업
- [ ] 결과 분석
- [ ] 파일럿 성공 여부 판단

### 분석 항목
1. **생성 품질**
   - 각 위반 유형이 명확한가?
   - 자연스러운가?
   - Judge agreement (선택적)

2. **기술적 검증**
   - 데이터 누수 없는가?
   - 토큰 길이 문제 없는가?
   - 파이프라인 안정적인가?

3. **모델 성능**
   - Macro F1 > 0.3?
   - Normal vs Violation 구분되는가?

### 결정
**✅ 성공 → 전체 600개로 확장**
- `configs/full.yaml` 작성
- 스크립트 재사용
- 3-4시간 생성

**❌ 실패 → 문제 진단 및 수정**
- 프롬프트 수정
- 설계 조정
- 파일럿 재실행

---

## 📊 각 단계 예상 시간

| Phase | 작업 | 시간 | 누적 |
|-------|------|------|------|
| 0 | 준비 | 5분 | 5분 |
| 1 | 설정 파일 | 10분 | 15분 |
| 2 | 프롬프트 | 30분 | 45분 |
| 3 | Normal 준비 | 20분 | 1h 5m |
| 4 | V1~V3 생성 | 1시간 | 2h 5m |
| 5 | V4~V5 생성 | 1시간 | 3h 5m |
| 6 | 품질 검증 | 1시간 | 4h 5m |
| 7 | Split | 10분 | 4h 15m |
| 8 | 입력 생성 | 1시간 | 5h 15m |
| 9 | 학습 | 30분 | 5h 45m |
| 10 | 분석 | 30분 | 6h 15m |

**총 예상 시간: 6-7시간 (1-2일)**

---

## 🚀 시작 방법

**지금부터 Phase 1부터 순차적으로 진행합니다.**

각 Phase 완료 후:
1. 검증 실행
2. 통과 조건 확인
3. 문제 있으면 수정
4. 다음 Phase로

**Phase 1 (설정 파일) 시작하시겠어요?**
