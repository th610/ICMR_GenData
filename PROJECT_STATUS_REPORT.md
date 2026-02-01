# ESConv Violation Detector - 프로젝트 현황 보고서

**작성일**: 2026-01-31  
**프로젝트**: ESConv 데이터셋 기반 감정지원 상담 대화 위반 탐지 시스템

---

## 📋 프로젝트 개요

### 목표
ESConv 감정지원 상담 데이터셋을 활용하여, 상담자의 부적절한 응답을 자동으로 탐지하는 6-class 분류 모델 개발

### 클래스 정의
1. **Normal**: 적절한 상담 응답
2. **V1 (맥락 파악 실패)**: 정보 부족 시점에 성급한 조언/해석/단정
3. **V2 (주체성 침해)**: 사용자 선택권 제한, 명령조 행동 지시
4. **V3 (저품질 공감)**: 형식적 공감, 표면적 응답으로 대화 진전 없음
5. **V4 (현실 왜곡)**: 비현실적/해로운 신념을 사실처럼 수용/강화
6. **V5 (위기 대응 실패)**: 자해/자살 신호에 안전 확인 없음

### 데이터 파이프라인
```
ESConv.json (1,300 sessions)
  ↓
Pilot: 30 sessions (각 클래스별 5개)
  ↓
Train/Val/Test Split
  ↓
모델 학습 (DistilRoBERTa-base 예정)
  ↓
평가 및 분석
```

---

## ✅ 완료된 작업 (Phase 1-6)

### **Phase 1: 프로젝트 설정 및 구조 정의**
- ✅ 6가지 위반 유형 정의 완료
- ✅ 데이터 구조 설계 (session_id, situation, dialog, label)
- ✅ configs/pilot.yaml 설정 파일 생성
- ✅ 프로젝트 디렉토리 구조 설정

**위치**: 
- `PHASE_GUIDE.md`: 전체 페이즈 가이드
- `PILOT_PLAN.md`: Pilot 30개 세션 계획
- `PIPELINE.md`: 전체 파이프라인 문서

---

### **Phase 2: Prompt 설계**
- ✅ Normal 생성 프롬프트 (대화 절단 방식)
- ✅ V1, V2, V3 생성 프롬프트 (재작성 방식)
- ✅ V4, V5 생성 프롬프트 (신규 생성 방식)
- ✅ 요약 생성 프롬프트 (LLM Judge용)

**위치**: 
- `src/llm/prompts_v2.py`: 모든 생성 프롬프트
- `src/llm/summarize.py`: 요약 생성 함수

---

### **Phase 3: 코드 구현**
- ✅ Normal 생성: `src/generation/normal_cutter.py`
- ✅ V1-V3 재작성: `src/generation/v1_v3_rewriter.py`
- ✅ V4-V5 생성: `src/generation/v4_v5_generator.py`
- ✅ OpenAI 클라이언트: `src/llm/openai_client.py`
- ✅ 유틸리티: `src/utils.py`

**실행 스크립트**:
- `scripts/pilot/step1_prepare_normal.py`
- `scripts/pilot/step2_generate_v1_v3.py`
- `scripts/pilot/step3_generate_v4_v5.py`

---

### **Phase 4: Pilot 30개 데이터 생성**
- ✅ Normal: 5개 (ESConv 원본에서 중간 절단)
- ✅ V1: 5개 (Normal 기반 재작성)
- ✅ V2: 5개 (Normal 기반 재작성)
- ✅ V3: 5개 (Normal 기반 재작성)
- ✅ V4: 5개 (신규 생성)
- ✅ V5: 5개 (신규 생성)

**위치**: `data/pilot/` (normal.json, v1.json, v2.json, v3.json, v4.json, v5.json)

**검증**:
- ✅ Phase 1 검증: `scripts/pilot/verify_phase1.py` (Normal 5개 생성 확인)
- ✅ Phase 2 검증: `scripts/pilot/verify_phase2.py` (V1-V5 생성 확인)

---

### **Phase 5: 데이터 구조 통일**
- ✅ 모든 클래스 동일한 JSON 구조로 저장
- ✅ session_id, situation, dialog, label 필드 표준화
- ✅ speaker: "seeker" / "supporter" 통일

---

### **Phase 6: 데이터 검증**
**6-1. 구조적 검증 (완료)**
- ✅ `scripts/pilot/validate_data.py` 구현
- ✅ 30개 세션 모두 통과
- ✅ 필드 존재, 턴 수, speaker 교대 확인

**6-2. LLM Judge 품질 검증 (진행 중 - 문제 발견)**
- ✅ `scripts/pilot/judge_quality.py` 구현
- ✅ Summary+Window 방식 (마지막 4턴 + 이전 대화 요약)
- ✅ gpt-4o-mini 사용, temperature=0.3
- ⚠️ **V3 과잉 판정 문제 발견**

---

## 🔴 발견된 주요 문제

### **문제 1: LLM Judge의 V3 과잉 판정**

#### **현상**
Pilot 30개 데이터 Judge 평가 결과 (최종):
```
Overall Accuracy: 36.7% (11/30)

Normal: 20% (1/5)  → 4개가 V3로 오판
V1:     0%  (0/5)  → 5개가 V3로 오판
V2:     0%  (0/5)  → 5개가 V3로 오판
V3:     100% (5/5) → 정확
V4:     0%  (0/5)  → 5개가 V3로 오판
V5:     100% (5/5) → 정확
```

**19개 중 19개가 모두 V3로 오판됨!**

#### **시도한 해결책들**
1. **프롬프트 개선 시도 1**: V3 적용 제외 조건 추가
   - 결과: ESConv 원본 80% Normal 인정 (개선)
   - 하지만 Pilot 데이터는 여전히 20% Normal만 인정

2. **프롬프트 개선 시도 2**: V3 기준을 4조건 모두 만족으로 강화
   - 결과: 오히려 악화 (50% → 10% Normal 인정)
   - Judge가 "사용자 감정 드러남" = "개입 필요"로 과도 해석

3. **프롬프트 개선 시도 3**: "명시적 도움 요청" 조건 추가 (매우 제한적)
   - 결과: V4까지 V3로 오판 (36.7% 정확도)
   - 거의 모든 seeker 발화를 "도움 요청"으로 해석

#### **근본 원인 분석**
- **가설 1**: 생성된 데이터가 실제로 품질이 낮음 (V3 수준)
  - Normal, V1, V2, V4로 생성했지만 실제로는 "일반적 조언", "형식적 공감"만 포함
  - 생성 프롬프트가 "깊이 있는 상담" 생성 실패

- **가설 2**: ESConv 원본 자체가 그 정도 품질
  - ESConv 원본 평가: 80% Normal (20% V3 위반)
  - 크라우드소싱 데이터라 전문 상담사 수준 아님
  - 우리가 생성한 데이터가 ESConv보다 나을 필요 없을 수도

- **가설 3**: V3 정의 자체가 너무 광범위
  - "더 잘할 수 있었음" vs "하지 않으면 안 되는 개입 실패"의 경계 모호
  - Judge가 과도하게 엄격하게 판단

---

### **문제 2: V1 vs V3 구분 실패**
- V1 5개 모두 V3로 판정됨
- 이유: "정보 부족 시점의 성급한 판단"과 "정보 충분하나 깊이 없는 대응"의 구분 실패
- Judge가 "정보가 충분히 드러났다"고 거의 항상 판단

---

### **문제 3: V2 vs V3 구분 실패**
- V2 5개 모두 V3로 판정됨
- 이유: "주체성 침해"보다 "품질 저하"가 더 넓은 개념으로 해석됨

---

### **문제 4: V4 vs V3 구분 실패**
- V4 5개 중 0~5개가 V3로 판정됨 (프롬프트에 따라 변동)
- 이유: "현실 왜곡" 응답도 "형식적 공감" 범주로 흡수됨

---

## 📊 ESConv 원본 품질 분석

### **ESConv 원본 20개 샘플 평가 결과**
(개선된 프롬프트 기준)
```
Normal: 16개 (80%)
V3:     4개  (20%)
V1, V2, V4, V5: 0개
```

**V3 위반 사례**:
1. "no problem you can do it!" (임신 고민에 단순 응원)
2. 운동 추천만 (건강 불안에 감정 탐색 없음)
3. "Ok thanks" (시스템 메시지 직후 형식적 종료)
4. "Okay thank you for your time" (실업 문제에 감정 탐색 없음)

**종료 인사 Normal 인정 (개선 성공)**:
- "You're welcome", "Sure, my pleasure!", "Take care", "Have a wonderful holiday" 등

---

## 🎯 전체 파이프라인 단계

### **Phase 1-6: Pilot 데이터 생성 및 검증** ✅ (문제 발견)
- [x] Phase 1: 프로젝트 설정
- [x] Phase 2: Prompt 설계
- [x] Phase 3: 코드 구현
- [x] Phase 4: Pilot 30개 생성
- [x] Phase 5: 데이터 구조 통일
- [x] Phase 6: 검증 (구조 ✅, 품질 ⚠️)

### **Phase 7: Train/Val/Test Split** ⏸️ (보류)
- [ ] 30개 → 24/3/3 (80/10/10) 분할
- [ ] Stratified by class
- [ ] `scripts/pilot/split_data.py`

### **Phase 8: 전처리 및 Tokenizer** ⏸️ (보류)
- [ ] DistilRoBERTa-base tokenizer
- [ ] Summary+Window 입력 형식
- [ ] `scripts/pilot/preprocess.py`

### **Phase 9: Pilot 모델 학습** ⏸️ (보류)
- [ ] Feasibility test (30개 데이터)
- [ ] 모델: DistilRoBERTa-base
- [ ] Loss, Accuracy, F1 확인
- [ ] `scripts/step6_train.py` (pilot 버전)

### **Phase 10: Pilot 평가** ⏸️ (보류)
- [ ] Confusion matrix
- [ ] Class-wise F1
- [ ] 샘플 분석
- [ ] `scripts/step7_evaluate.py` (pilot 버전)

### **Phase 11: Scale-up 결정** ⏸️ (보류)
- [ ] Pilot 결과 분석
- [ ] 전체 데이터 생성 여부 결정
- [ ] (300-500 sessions per class)

### **Phase 12-14: 전체 파이프라인** ⏸️ (미진행)
- [ ] 전체 데이터 생성
- [ ] 모델 재학습
- [ ] 최종 평가

---

## 🔍 현재 상태 요약

### **완료된 것**
1. ✅ Pilot 30개 세션 생성 완료
2. ✅ 구조적 검증 통과
3. ✅ LLM Judge 시스템 구축
4. ✅ ESConv 원본 품질 분석 (80% Normal)
5. ✅ V3 정의 여러 차례 개선 시도

### **막힌 것**
1. ❌ Pilot 데이터의 LLM Judge 품질 검증 실패 (36.7% 정확도)
2. ❌ V3 과잉 판정으로 다른 클래스와 구분 불가
3. ❌ V1, V2, V4 모두 V3로 흡수되는 현상

### **원인 불명확**
- 생성 데이터가 실제로 품질이 낮은가?
- Judge가 너무 엄격한가?
- V3 정의가 너무 광범위한가?
- ESConv 기반이라 애초에 한계가 있는가?

---

## 💡 다음 조치 옵션

### **옵션 A: 데이터 실제 확인 (추천)**
1. Normal 1개 (normal_0000 - 유일하게 통과) 내용 확인
2. V3 5개 (모두 정확히 판정) 내용 확인
3. V3로 오판된 Normal/V1/V2/V4 샘플 확인
4. 실제로 생성 품질 문제인지 Judge 문제인지 판단

### **옵션 B: V3 정의 재검토**
1. V3를 더 좁은 범위로 재정의 (예: 심각한 품질 문제만)
2. V1, V2, V4와 명확히 구분되는 기준 마련
3. "더 잘할 수 있었음" vs "명백한 역할 실패" 경계 명확화

### **옵션 C: 생성 프롬프트 개선**
1. Normal, V1, V2 생성 시 "깊이 있는 상담" 강조
2. "일반적 조언" 지양, "구체적 감정 탐색" 강조
3. 재생성 후 Judge 재평가

### **옵션 D: Judge 기준 완화**
1. V3를 "참고용 품질 지표"로만 사용
2. V1, V2, V4, V5만 명확히 구분
3. Normal vs V3는 모델이 학습으로 구분하도록 위임

### **옵션 E: Pilot 포기하고 규칙 기반 생성**
1. LLM 생성 대신 규칙 기반으로 명확한 위반 생성
2. V1: 초반 3턴 이내 조언
3. V2: "You should", "You must" 강제 삽입
4. V3: 마지막 5턴 모두 "I see", "That's good" 반복
5. V4: 명시적 거짓 정보 삽입
6. V5: "자살" 언급에 "힘내세요"만 응답

---

## 📁 주요 파일 위치

### **설정 및 문서**
- `configs/pilot.yaml`: Pilot 설정
- `PHASE_GUIDE.md`: 전체 가이드
- `PILOT_PLAN.md`: Pilot 계획
- `PIPELINE.md`: 파이프라인 문서
- `PROGRESS_SUMMARY.txt`: 진행 요약

### **생성 코드**
- `src/generation/normal_cutter.py`: Normal 생성
- `src/generation/v1_v3_rewriter.py`: V1-V3 재작성
- `src/generation/v4_v5_generator.py`: V4-V5 생성
- `src/llm/prompts_v2.py`: 모든 프롬프트
- `src/llm/summarize.py`: 요약 생성

### **실행 스크립트**
- `scripts/pilot/step1_prepare_normal.py`: Normal 5개 생성
- `scripts/pilot/step2_generate_v1_v3.py`: V1-V3 각 5개 생성
- `scripts/pilot/step3_generate_v4_v5.py`: V4-V5 각 5개 생성

### **검증 스크립트**
- `scripts/pilot/validate_data.py`: 구조 검증
- `scripts/pilot/judge_quality.py`: LLM Judge (Pilot 30개)
- `scripts/pilot/judge_esconv_original.py`: ESConv 원본 평가
- `scripts/pilot/verify_phase1.py`: Phase 1 검증
- `scripts/pilot/verify_phase2.py`: Phase 2 검증

### **데이터**
- `data/pilot/normal.json`: Normal 5개
- `data/pilot/v1.json`: V1 5개
- `data/pilot/v2.json`: V2 5개
- `data/pilot/v3.json`: V3 5개
- `data/pilot/v4.json`: V4 5개
- `data/pilot/v5.json`: V5 5개
- `data/pilot/judge_results_summary_window.json`: Judge 결과
- `data/pilot/judge_esconv_original_20.json`: ESConv 평가 결과

### **원본 데이터**
- `ESConv.json`: 1,300개 원본 세션

---

## 🎯 권장 다음 단계

### **즉시 수행**
1. **데이터 실제 확인** (옵션 A)
   - normal_0000 (통과) 내용 읽기
   - v3_0000~0004 (정확) 내용 읽기
   - normal_0001 (V3로 오판) 내용 읽기
   - 실제 품질 문제인지 확인

2. **판단 기준**
   - 생성 데이터가 정말 품질이 낮다면 → 옵션 C (프롬프트 개선)
   - Judge가 너무 엄격하다면 → 옵션 B (V3 재정의)
   - 둘 다 애매하다면 → 옵션 D (Judge 완화) 또는 E (규칙 기반)

### **중장기 방향**
- V3 문제 해결 후 Phase 7 진행 (Train/Val/Test Split)
- Pilot 모델 학습으로 feasibility 확인
- 성공 시 전체 데이터 생성 (Phase 11)

---

## 📈 성공 지표

### **Pilot 단계 성공 기준**
- [ ] LLM Judge 정확도 70% 이상 (현재 36.7%)
- [ ] 각 클래스별 정확도 50% 이상
- [ ] V5 (위기) 90% 이상 유지 (현재 100% ✅)
- [ ] V4 (현실왜곡) 70% 이상 (현재 0%)

### **모델 학습 성공 기준**
- [ ] Overall F1 > 0.6
- [ ] V5 Recall > 0.9 (위기 놓치면 안 됨)
- [ ] Normal vs V3 구분 F1 > 0.5

---

## 🔧 기술 스택

- **LLM**: gpt-4o-mini (생성 및 Judge)
- **모델 예정**: DistilRoBERTa-base
- **프레임워크**: PyTorch (예정)
- **언어**: Python 3.13
- **데이터**: ESConv.json (1,300 sessions)
- **평가**: LLM Judge + 구조 검증

---

## 📝 마지막 업데이트

- **날짜**: 2026-01-31
- **마지막 작업**: V3 과잉 판정 문제 분석 완료
- **현재 막힌 지점**: Pilot 데이터 품질 검증 실패 (36.7% 정확도)
- **다음 액션**: 데이터 실제 내용 확인 후 방향 결정

---

**이 보고서는 새로운 세션에서 프로젝트 상태를 빠르게 파악하기 위한 문서입니다.**
