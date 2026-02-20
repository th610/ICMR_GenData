# VACCINE: Violation-Aware Containment and Control for Interactional Norm Enforcement

LLM 기반 상담 시스템에서 발생하는 5가지 안전 위반(V1–V5)을 실시간 탐지 · 교정하는 경량 파이프라인

## Violation Taxonomy

| Type | Name | Description |
|------|------|-------------|
| **V1** | Context Understanding Failure | 맥락/정보 수집 없이 성급한 결론 |
| **V2** | User Autonomy Violation | 내담자 자율성 침해, 지시/강제 |
| **V3** | Superficial Empathy | 피상적·진부한 공감 표현 |
| **V4** | Reality Distortion | 비현실적 믿음 강화, 현실 왜곡 |
| **V5** | Crisis Safety Failure | 위기 상황 부적절 대응 |

## Pipeline Architecture

```
User Input → GPT-4o-mini (Response Gen)
                  ↓
           RoBERTa Detector (29ms)
              ↓          ↓
          Normal      Violation detected
           ↓               ↓
        Deliver     LLM Rewrite (max 3 attempts)
                         ↓
                   Re-detect → Safe? → Deliver
                         ↓
                      Failsafe (generic safe response)
```

## Project Structure

```
src/
├── agent/                    # 6-step agent pipeline
│   ├── step1_context_builder.py
│   ├── step2_candidate_generator.py
│   ├── step3_violation_detector.py
│   ├── step4_controller.py
│   ├── step5_logger.py
│   └── step6_agent.py
├── llm/                      # LLM modules
│   ├── openai_client.py      # API wrapper
│   ├── prompts.py            # V1–V5 generation prompts
│   ├── prompts_judge.py      # Judge classification prompts
│   ├── judge.py              # LLM-Judge evaluator
│   └── summarize.py          # Dialogue summarizer
├── synth/
│   └── rewrite_turn.py       # Violation rewriting
└── utils.py

generation/                   # Data generation pipeline
├── 01_assign_sessions.py     # ESConv session assignment (1,300)
├── 02_generate_violations.py # V1–V5 violation generation
├── 03_generate_normal.py     # Normal response generation
├── 03_merge_all_labels.py    # Merge all labels → 1,300
├── 04_evaluate_quality.py    # LLM-Judge quality evaluation
├── 04_split_exact.py         # Stratified train/test split
└── 05_split_dataset.py       # Final dataset split

models/
├── model.py                  # RoBERTa classifier definition
├── data_utils.py             # Data loading utilities
├── train.py                  # Training script
└── data/                     # Train/test/valid splits

data/
├── external/                 # Source datasets (ESConv, etc.)
└── final/                    # Final generated dataset
    ├── all_labels_1300.json  # Full 1,300 samples
    ├── train_1000.json       # Training set (1,000)
    └── test_gold_300.json    # Gold test set (300)

experiments/
├── paper_targets/            # Paper table scripts (editable DATA dicts)
│   ├── table1_safety.py      # Table 1: Structural Safety Guarantee
│   ├── table2_gate.py        # Table 2: Gate Signal Comparison
│   ├── table3_cost.py        # Table 3: Cost/Latency Tradeoff
│   ├── table4_detector.py    # Table 4: Detector Performance
│   ├── table5_natural_rate.py # Table 5: Natural Violation Rate
│   ├── run_all.py            # Run all tables
│   └── figures/              # Generated figures (PDF/PNG)
├── ablation/                 # Ablation study
├── figures/                  # Experiment figures
├── table1/ ~ table6_swmh/   # Raw experiment results
└── per_sample/               # Per-sample detailed results (300)

configs/                      # YAML configurations
csv_edit/                     # Manual quality review CSVs (train)
csv_edit_gold/                # Manual quality review CSVs (gold)
```

## Key Results

### Detector (RoBERTa v3, n=300)
- **Accuracy**: 93.33% (280/300)
- **Macro F1**: 0.9402
- **FNR**: 3.18% (7/220 violations missed)
- **Gate latency**: 29.1ms

### Safety Pipeline (n=220 violations)
- **Detection**: 213/220 (96.8%)
- **Correction**: 201/213 rewritten successfully
- **Failsafe**: 12/213 → generic safe response
- **Post-pipeline leakage**: 0 (structural guarantee)

### vs LLM-Judge Baseline
| Metric | VACCINE | LLM-Judge |
|--------|---------|-----------|
| Accuracy | 93.33% | 82.33% |
| Gate latency | 29.1ms | 878.8ms |
| FNR | 3.18% | 8.18% |
| E[Cost] | 8,056ms | 2,372ms |

**Breakeven**: VACCINE is cost-effective when violation rate > 9.7%

## Setup

```bash
pip install -r requirements.txt
```

### Training
```bash
python models/train.py
```

### Paper Tables
```bash
cd experiments/paper_targets
python run_all.py          # Generate all tables + figures
python table1_safety.py    # Individual table
```

## Data

- **Source**: ESConv (1,300 sessions)
- **Generation**: GPT-4o-mini with violation-specific prompts
- **Split**: 1,000 train / 300 test (stratified)
- **Labels**: Normal (80) + V1 (50) + V2 (60) + V3 (30) + V4 (40) + V5 (40) in test

## License

MIT
