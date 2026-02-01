# Violation-aware Dialogue Agent

## 구조

```
Context Builder → Candidate Generator → Violation Detector → Controller → Final Response
                                                    ↓
                                                  Logger
```

## 설치 및 실행

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. 모델 학습 (옵션)

Detector를 학습된 모델로 사용하려면:

```bash
python scripts/train_classifier.py
```

Judge 모드(LLM 평가)를 사용하려면 학습 불필요.

## 사용 예제

### Interactive Mode

```bash
python demo_agent.py --mode interactive
```

### Single Turn Demo

```bash
python demo_agent.py --mode single
```

### Violation Detection Test

```bash
python demo_agent.py --mode violation
```

## 구성 요소

### 1. Context Builder
- **역할**: 대화 히스토리 압축
- **출력**: 요약 + 최근 N턴
- **파라미터**:
  - `window_size`: 최근 턴 개수 (기본 6)
  - `use_summary`: 긴 대화 요약 사용 여부

### 2. Candidate Generator
- **역할**: K개 후보 응답 생성
- **출력**: 다양한 응답 후보들
- **파라미터**:
  - `num_candidates`: 후보 개수 K (기본 3)
  - `temperature`: 다양성 (기본 0.8)

### 3. Violation Detector
- **역할**: 위반 패턴 감지
- **모드**:
  - `model`: 학습된 분류기 (빠름, 일관성)
  - `judge`: LLM 평가 (느림, 유연성)
- **출력**:
  - Multi-label violations (V1-V5)
  - Top violation
  - Confidence
  - Evidence

### 4. Controller
- **역할**: 최종 응답 결정
- **정책**:
  - **NONE**: 위반 없음 → 그대로 선택
  - **SOFT**: 일반 위반 → 최선 선택 또는 수정
  - **HARD**: 고위험 → 안전 템플릿 강제
- **파라미터**:
  - `enable_modification`: 수정 기능 활성화

### 5. Logger
- **역할**: 추적 및 디버깅
- **출력**:
  - 콘솔: 요약 정보
  - 파일: 상세 JSON 로그
  - 통계: 세션 전체 분석

## 위반 타입

- **V1: Empathy Violation** - 감정 무시, 단정
- **V2: Fact-Checking Violation** - 사실 확인 없이 주장
- **V3: Advice Violation** - 중요한 결정에 직접 조언
- **V4: Safety Violation (Mixed)** - 복합 위험 패턴
- **V5: Safety Violation (Single)** - 심각한 단일 위반

## 커스터마이징

### 후보 개수 변경

```python
candidate_generator = CandidateGenerator(
    llm_client=llm,
    num_candidates=5,  # K=5로 증가
    temperature=0.9
)
```

### 안전 템플릿 수정

`src/agent/controller.py`의 `_get_safety_template()` 수정

### 새로운 정책 추가

`src/agent/controller.py`의 `InterventionPolicy` 클래스 확장

## 로그 분석

```python
# 로그 읽기
import json
log = json.load(open('logs/agent/session_xxx.json'))

# 통계 확인
print(log['statistics'])

# 특정 턴 분석
turn_0 = log['turns'][0]
print(turn_0['detections'])
```

## 성능 고려사항

- **Judge 모드**: 정확하지만 느림 (턴당 ~10초)
- **Model 모드**: 빠르지만 학습 필요 (턴당 ~1초)
- **후보 개수**: K↑ = 품질↑ but 비용↑

## 다음 단계

1. ✅ 기본 구조 완성
2. ⏭️ 모델 학습 및 평가
3. ⏭️ 정책 최적화
4. ⏭️ 실전 데이터 수집
5. ⏭️ A/B 테스트
