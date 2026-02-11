# 프롬프트 비교: 원래 Normal 생성 vs 증강 Normal 생성

## 1. 목적 차이

### 원래 Normal 생성 (prompts.py - NORMAL_SYSTEM)
**목적**: ESConv prefix 뒤에 **새로운 4턴 전체**를 Normal로 생성  
**Input**: prefix_dialog만 (12턴)  
**Output**: 완전히 새로운 4턴 (Turn 1, 2, 3, 4 모두 생성)

### 증강 Normal 생성 (prompts_augmentation.py)
**목적**: 위반 샘플의 **Turn 4만** Normal로 교체  
**Input**: prefix_dialog + 이미 생성된 Turn 1-3  
**Output**: Turn 4 하나만 생성 (Turn 1-3은 원본 유지)

---

## 2. 프롬프트 구조 비교

### 원래 Normal 생성
```
[Input]
ESConv Prefix (Turn 0 ~ Turn 12): [prefix만]

[Task]
Generate EXACTLY 4 turns after prefix

Turn 1 (seeker): Continue conversation naturally
Turn 2 (supporter): Good supportive response
Turn 3 (seeker): Respond / add detail
Turn 4 (supporter): Good supportive response

[Requirements]
- Continue same topic from prefix
- Reuse ONE entity from prefix
- Turn 1 must connect with continuation cues
- Turn 4 FORBIDDEN: V1-V5 violations

[Output]
{
  "dialog": [
    {"speaker": "seeker", "content": "..."},
    {"speaker": "supporter", "content": "..."},
    {"speaker": "seeker", "content": "..."},
    {"speaker": "supporter", "content": "..."}
  ],
  "primary_label": "Normal"
}
```

### 증강 Normal 생성 (V1 예시)
```
[Input - ESConv Prefix Dialogue]
Situation: ...
[전체 prefix_dialog 표시]

[Input - Already Generated Dialogue]
Turn 0 (seeker): [이미 생성됨]
Turn 1 (supporter): [이미 생성됨]
Turn 2 (seeker): [이미 생성됨]

[Task]
Generate ONLY Turn 4 (last supporter response)

[IMPORTANT CONTEXT]
- Turn 3 contains UNCERTAINTY
- Seeker hasn't confirmed facts
- NO V1 violation

[Turn 4 Requirements - Avoid V1 Violation]
✅ DO:
  - Ask ONE clarifying question
  - Acknowledge uncertainty
  - Gentle next steps WITHOUT assumptions

❌ DON'T (V1 patterns):
  - "Even without knowing all the details..."
  - "It seems clear that..."
  - Jump to conclusions

[Output]
{
  "supporter_response": "..."
}
```

---

## 3. 핵심 차이점

| 항목 | 원래 Normal 생성 | 증강 Normal 생성 |
|------|-----------------|-----------------|
| **생성 범위** | 4턴 전체 (Turn 1-4) | Turn 4 하나만 |
| **맥락 정보** | prefix만 | prefix + Turn 1-3 |
| **Turn 1-3** | 새로 생성 | 원본 그대로 유지 |
| **목적** | 처음부터 Normal 대화 생성 | 위반 Turn 4를 Normal로 교체 |
| **출력 형식** | `{"dialog": [...]}` | `{"supporter_response": "..."}` |
| **위반 회피 지침** | 일반적 금지 사항 | V1-V5별 specific 패턴 명시 |
| **맥락 연결** | Turn 1이 prefix 연결 | Turn 4가 Turn 3 이어받음 |

---

## 4. 위반별 맞춤 지침 (증강만 해당)

### V1 증강 (성급한 결론 회피)
```
✅ DO: Ask clarifying question, acknowledge uncertainty
❌ DON'T: "Even without all details...", jump to conclusions
```

### V2 증강 (강요 회피)
```
✅ DO: Offer options (A/B), ask preference, support autonomy
❌ DON'T: "You must...", "You have to...", commands
```

### V3 증강 (실질적 도움 제공)
```
✅ DO: Empathy + concrete help (question/options/guide)
❌ DON'T: Empathy-only when practical help requested
```

### V4 증강 (극단적 믿음 질문)
```
✅ DO: Validate feeling, reality-check question, alternatives
❌ DON'T: Confirm extreme beliefs as fact
```

### V5 증강 (안전 프로토콜)
```
✅ DO: Safety question, emergency resources (1393/112), immediate help
❌ DON'T: Generic empathy without safety check
```

---

## 5. 실제 예시 비교

### 원래 Normal 생성 결과
```json
{
  "dialog": [
    {"speaker": "seeker", "content": "Also, I've been thinking about what we discussed..."},
    {"speaker": "supporter", "content": "I'm glad you're reflecting on that..."},
    {"speaker": "seeker", "content": "Yes, it's been helpful to talk..."},
    {"speaker": "supporter", "content": "What feels like the most important next step for you?"}
  ],
  "primary_label": "Normal"
}
```
→ **4턴 모두 새로 생성됨**

### 증강 Normal 생성 결과
```json
// 입력 (원본 위반 샘플)
{
  "generated_dialog": [
    {"speaker": "seeker", "text": "Also, something else happened..."},  // 유지
    {"speaker": "supporter", "text": "That sounds concerning..."},      // 유지
    {"speaker": "seeker", "text": "I can't tell for sure..."},          // 유지
    {"speaker": "supporter", "text": "Even without knowing all the details, it seems clear..."}  // 🔴 V1 위반
  ]
}

// 출력 (증강 샘플)
{
  "generated_dialog": [
    {"speaker": "seeker", "text": "Also, something else happened..."},  // ✅ 동일
    {"speaker": "supporter", "text": "That sounds concerning..."},      // ✅ 동일
    {"speaker": "seeker", "text": "I can't tell for sure..."},          // ✅ 동일
    {"speaker": "supporter", "text": "What specifically made you uncertain? Can you tell me more about what you heard?"}  // ✅ Normal로 교체
  ],
  "primary_label": "Normal",
  "augmentation_meta": {
    "original_label": "V1",
    "original_turn4": "Even without knowing all the details, it seems clear..."
  }
}
```
→ **Turn 4만 교체, Turn 1-3은 원본 유지**

---

## 6. 왜 이렇게 바꿨나?

### ✅ 장점 1: Contrastive Learning
- 같은 대화 맥락 (prefix + Turn 1-3)
- 다른 응답 (Turn 4: 위반 vs Normal)
- 모델이 "무엇이 위반인지" 명확히 학습

### ✅ 장점 2: 자연스러운 생성
- GPT가 원본 위반 응답을 안 봄 (프롬프트에 없음)
- Turn 3 맥락만 보고 자연스럽게 이어감
- 위반 응답의 영향 받지 않음

### ✅ 장점 3: 원본 생성 방식과 일관성
- 같은 "Turn 4 생성" 태스크
- 같은 "violation 회피" 지침 스타일
- 기존 데이터와 자연스럽게 섞임

### ✅ 장점 4: 메타데이터 추적
- augmentation_meta로 원본 정보 보존
- 원본 위반 응답 기록
- 디버깅 및 분석 가능

---

## 7. 데이터셋 구조 변화

### Before (train_1000.json)
```
- Normal: 445개
- V1-V5: 555개
Total: 1,000개
```

### After (train_1600_augmented.json)
```
- Normal: 1,000개 (445 원본 + 555 증강)
- V1-V5: 555개 (원본 유지)
Total: 1,555개

Session IDs:
- 원본: session_1 ~ session_1000
- 증강: augmented_1301 ~ augmented_1855
```

---

## 8. 예상 효과

1. **데이터 증가**: 1000 → 1555 (55% 증가)
2. **파라미터/샘플 비율**: 125M/800 = 156,250:1 → 125M/1555 = 80,386:1 (절반!)
3. **Normal 밸런스**: 44.5% → 64.3% (Normal 비중 증가)
4. **Contrastive pairs**: 555쌍 (같은 맥락, 다른 응답)
5. **Overfitting 완화**: 테스트 정확도 100% → 85-90% 예상
