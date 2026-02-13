#!/usr/bin/env python3
"""
Shortcut / Leakage 진단 스크립트
=================================
현재 학습된 RoBERTa 모델이 진짜 대화 품질을 보고 판단하는지,
Shortcut이나 Leakage로 정답을 맞추는지 테스트.

실험 1: Response-only  → context(prefix + generated Turn 1~3) 제거, Turn 4(TARGET)만 입력
         → 성능이 그대로 높으면 shortcut (응답 텍스트만으로 레이블 구분 가능)

실험 2: Context-only   → Turn 4(TARGET) 제거, prefix + generated Turn 1~3만 입력
         → 성능이 높으면 prefix leakage (context만으로 레이블 추론 가능)

사용법:
    python test_shortcut_diagnosis.py
"""

import os
import sys
import json
import torch
import numpy as np
from collections import Counter
from transformers import RobertaTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Add models/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from model import ViolationClassifier
from data_utils import DataConfig, setup_tokenizer

# ============================================================================
# Config
# ============================================================================
import sys
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/outputs/best_model.pt"
TEST_DATA_PATH = "data/final/test_gold_300.json"
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
BATCH_SIZE = 16
LABELS = DataConfig.LABELS  # ["Normal", "V1", "V2", "V3", "V4", "V5"]
LABEL2ID = DataConfig.LABEL2ID
ID2LABEL = DataConfig.ID2LABEL


# ============================================================================
# Text Formatting Functions (3 modes)
# ============================================================================

def format_full(sample, tokenizer):
    """정상 모드: 원래 학습 시 사용한 역방향 truncation 그대로 적용"""
    from data_utils import format_conversation_with_truncation
    prefix_dialog = sample.get("prefix_dialog", [])
    generated_dialog = sample["generated_dialog"]
    return format_conversation_with_truncation(prefix_dialog, generated_dialog, tokenizer, MAX_LENGTH)


def format_response_only(sample, tokenizer):
    """실험 1: Response-only → Turn 4(TARGET)만 입력"""
    generated_dialog = sample["generated_dialog"]
    target_content = generated_dialog[3]["content"].strip()
    return f"{DataConfig.SUPPORTER_TARGET_TOKEN} {target_content}"


def format_context_only(sample, tokenizer):
    """실험 2: Context-only → prefix + Turn 1~3만, TARGET 제거"""
    prefix_dialog = sample.get("prefix_dialog", [])
    generated_dialog = sample["generated_dialog"]

    parts = []
    # prefix turns
    for turn in prefix_dialog:
        speaker = turn["speaker"].lower()
        token = DataConfig.SEEKER_TOKEN if speaker == "seeker" else DataConfig.SUPPORTER_TOKEN
        parts.append(f"{token} {turn['content'].strip()}")

    # generated Turn 1~3 (TARGET 제외)
    for turn in generated_dialog[:3]:
        speaker = turn["speaker"].lower()
        token = DataConfig.SEEKER_TOKEN if speaker == "seeker" else DataConfig.SUPPORTER_TOKEN
        parts.append(f"{token} {turn['content'].strip()}")

    return "\n".join(parts)


# ============================================================================
# Inference
# ============================================================================

def run_inference(model, tokenizer, samples, format_fn, device):
    """포맷 함수에 따라 모델 추론 실행"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch_samples = samples[i : i + BATCH_SIZE]

            texts = [format_fn(s, tokenizer) for s in batch_samples]
            labels = [LABEL2ID[s["primary_label"]] for s in batch_samples]

            encoding = tokenizer(
                texts,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_preds, all_labels


# ============================================================================
# Report
# ============================================================================

def print_report(name, preds, labels):
    """분류 리포트 출력"""
    acc = accuracy_score(labels, preds)
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro P:   {prec:.4f}")
    print(f"  Macro R:   {rec:.4f}")
    print(f"  Macro F1:  {f1:.4f}")

    # Per-class report
    target_names = [ID2LABEL[i] for i in range(len(LABELS))]
    print(f"\n  Per-class breakdown:")
    print(
        classification_report(
            labels, preds, target_names=target_names, digits=3, zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    header = "        " + "  ".join(f"{l:>6}" for l in LABELS)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {LABELS[i]:>6}  {row_str}")

    return {"accuracy": acc, "macro_f1": f1, "macro_precision": prec, "macro_recall": rec}


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load tokenizer + special tokens
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer, num_added = setup_tokenizer(tokenizer)
    print(f"  Added {num_added} special tokens: {DataConfig.SEEKER_TOKEN}, {DataConfig.SUPPORTER_TOKEN}, {DataConfig.SUPPORTER_TARGET_TOKEN}")

    # 2. Load model
    print("Loading model...")
    model = ViolationClassifier(
        model_name=MODEL_NAME,
        num_labels=len(LABELS),
        dropout=0.1,
        pooling="cls",
    )
    # Resize embeddings to match special tokens
    model.roberta.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"  Loaded from epoch {ckpt['epoch']}, best_metric={ckpt['best_metric']}")

    # 3. Load test data
    print("Loading test data...")
    with open(TEST_DATA_PATH, "r") as f:
        test_data = json.load(f)
    samples = test_data["samples"]
    print(f"  {len(samples)} samples")

    label_dist = Counter(s["primary_label"] for s in samples)
    print(f"  Distribution: {dict(sorted(label_dist.items()))}")

    # 4. Run 3 experiments
    results = {}

    # Baseline: Full input (sanity check)
    print("\n>>> Running: Full (baseline)...")
    preds_full, labels_full = run_inference(model, tokenizer, samples, format_full, device)
    results["full"] = print_report("Baseline: Full Input (prefix + Turn1~3 + TARGET)", preds_full, labels_full)

    # Experiment 1: Response-only
    print("\n>>> Running: Response-only...")
    preds_resp, labels_resp = run_inference(model, tokenizer, samples, format_response_only, device)
    results["response_only"] = print_report("Exp 1: Response-Only (TARGET만, context 제거)", preds_resp, labels_resp)

    # Experiment 2: Context-only
    print("\n>>> Running: Context-only...")
    preds_ctx, labels_ctx = run_inference(model, tokenizer, samples, format_context_only, device)
    results["context_only"] = print_report("Exp 2: Context-Only (prefix + Turn1~3, TARGET 제거)", preds_ctx, labels_ctx)

    # 5. Summary comparison
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: Shortcut / Leakage 진단 결과")
    print(f"{'=' * 70}")
    print(f"  {'Experiment':<35} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*55}")
    for key, label in [
        ("full",          "Full (baseline)"),
        ("response_only", "Response-Only (shortcut test)"),
        ("context_only",  "Context-Only (leakage test)"),
    ]:
        r = results[key]
        print(f"  {label:<35} {r['accuracy']:>9.1%} {r['macro_f1']:>9.3f}")

    print(f"\n  해석:")
    full_acc = results["full"]["accuracy"]
    resp_acc = results["response_only"]["accuracy"]
    ctx_acc = results["context_only"]["accuracy"]
    chance = 1.0 / len(LABELS)

    if resp_acc > 0.8 * full_acc:
        print(f"  ⚠️  Response-Only ({resp_acc:.1%}) ≈ Full ({full_acc:.1%})")
        print(f"      → 모델이 응답 텍스트만으로 분류. Shortcut 가능성 높음.")
    elif resp_acc > chance + 0.1:
        print(f"  ⚡ Response-Only ({resp_acc:.1%}) > chance ({chance:.1%})")
        print(f"      → 응답 텍스트에 약간의 레이블 단서가 있으나 context 의존도 있음.")
    else:
        print(f"  ✅ Response-Only ({resp_acc:.1%}) ≈ chance ({chance:.1%})")
        print(f"      → 응답만으로는 분류 불가. Shortcut 없음.")

    if ctx_acc > 0.8 * full_acc:
        print(f"  ⚠️  Context-Only ({ctx_acc:.1%}) ≈ Full ({full_acc:.1%})")
        print(f"      → 모델이 context만으로 분류. Prefix Leakage 가능성 높음.")
    elif ctx_acc > chance + 0.1:
        print(f"  ⚡ Context-Only ({ctx_acc:.1%}) > chance ({chance:.1%})")
        print(f"      → Context에 약간의 레이블 단서가 있으나 TARGET 의존도 있음.")
    else:
        print(f"  ✅ Context-Only ({ctx_acc:.1%}) ≈ chance ({chance:.1%})")
        print(f"      → Context만으로는 분류 불가. Leakage 없음.")

    # 6. Save results
    output_path = "shortcut_diagnosis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  결과 저장: {output_path}")


if __name__ == "__main__":
    main()
