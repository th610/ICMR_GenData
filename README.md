# ESConv Violation Detector PoC

상담사 응답에서 5가지 위반 패턴(V1-V5)을 자동 탐지하는 multi-label classifier

## 위반 유형
- **V1**: 맥락 파악 실패 (정보 수집 부족)
- **V2**: 내담자 주도권 침해
- **V3**: 저품질 공감 (진부한 표현)
- **V4**: 현실 왜곡
- **V5**: 위기 안전 실패

## 프로젝트 구조

```
configs/
  poc.yaml                    # 전체 설정

src/
  llm/                        # LLM 관련 모듈
    openai_client.py          # API wrapper
    prompts.py                # 프롬프트 템플릿
    summarize.py              # 대화 요약
    judge.py                  # LLM-judge 라벨링
  synth/                      # 합성 데이터
    rewrite_turn.py           # 위반 주입
  data/                       # 데이터 처리
    make_turn_samples.py      # Turn 샘플 생성
  utils.py                    # 유틸리티

scripts/
  step1_sample_original.py    # 원본 세션 샘플링
  step2_generate_synthetic.py # 합성 데이터 생성
  step3_split_sessions.py     # Train/Val/Test 분할
  step4_make_turn_samples.py  # Turn 샘플 생성
  step5_label_turns.py        # LLM 라벨링
  step6_train.py              # 모델 학습
  step7_evaluate.py           # 테스트 평가
```

## 실행 방법

### 1. 환경 설정
```bash
pip install transformers torch accelerate scikit-learn pyyaml openai
```

### 2. 설정 파일 수정
`configs/poc.yaml`에서 OpenAI API 키 및 파라미터 조정

### 3. 파이프라인 실행

```bash
# Step 1: 원본 50세션 샘플링
python scripts/step1_sample_original.py --input ESConv.json --output data/sessions_original_50.json --num_sessions 50 --seed 42

# Step 2: 합성 50세션 생성 (위반 주입)
python scripts/step2_generate_synthetic.py --input data/sessions_original_50.json --output data/sessions_synth_50.json --seed 42

# Step 3: Train/Val/Test 분할 (80/10/10)
python scripts/step3_split_sessions.py --original data/sessions_original_50.json --synthetic data/sessions_synth_50.json --output_dir data/splits --seed 42

# Step 4: Turn 샘플 생성 (200개)
python scripts/step4_make_turn_samples.py --input_dir data/splits --output_dir data/turn_samples --config configs/poc.yaml

# Step 5: LLM-judge 라벨링
python scripts/step5_label_turns.py --input_dir data/turn_samples --output_dir data/labeled --config configs/poc.yaml

# Step 6: 모델 학습
python scripts/step6_train.py --input_dir data/labeled --output_dir models/detector --config configs/poc.yaml

# Step 7: 테스트 평가
python scripts/step7_evaluate.py --input_dir data/labeled --model_dir models/detector --output_dir models/detector --config configs/poc.yaml
```

## 주요 결과

### 데이터
- 원본 50 + 합성 50 = 총 100 세션
- Train 160, Val 20, Test 20 turn 샘플

### 모델
- **distilroberta-base** multi-label classifier
- 입력: `[SITUATION][SUMMARY][CONTEXT][RESPONSE]`
- 출력: 5-dim sigmoid (V1-V5)

### 성능 (Test)
```
Label    Precision  Recall    F1      Support
V1       0.00       0.00      0.00    14
V2       0.05       1.00      0.10    1
V3       0.50       1.00      0.67    10
V4       0.00       0.00      0.00    0
V5       0.00       0.00      0.00    1

Micro F1: 0.56
Macro F1: 0.50
```

## 알려진 이슈

1. **V4/V5 데이터 부족**: 합성 주입했으나 LLM-judge가 인식 못함
2. **V1 학습 실패**: 라벨은 많으나(92/160) 모델이 0개 예측
3. **V2/V3 과검출**: Class imbalance 문제
4. **소규모 데이터**: 160 train 샘플로는 부족

## 개선 방향

- LLM-judge 프롬프트 개선
- 데이터 증량 (200+200 세션)
- Class weight 조정
- 전문가 라벨링

## License

MIT
