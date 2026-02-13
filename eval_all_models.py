#!/usr/bin/env python3
"""
모델별 Gold Test 평가 비교
===========================
수정된 test_gold_300.json 기준으로 모든 모델 평가
"""

import json, torch, sys, os
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, "models")
from model import ViolationClassifier
from data_utils import DataConfig, ViolationDataset

LABELS = DataConfig.LABELS
TEST_PATH = "data/final/test_gold_300.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "best_model.pt":            "원본 (edit 0개)",
    "best_model_edited.pt":     "v1 (edit 170개)",
    "best_model_v2.pt":         "v2 (edit 265개)",
    "best_model_v3.pt":         "v3 (edit 979개)",
}


def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(ids, mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_labels) * 100

    cc, ct = defaultdict(int), defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        ct[l] += 1
        if p == l: cc[l] += 1

    per_class = {}
    for lid, lname in enumerate(LABELS):
        t = ct.get(lid, 0)
        per_class[lname] = (cc.get(lid, 0) / t * 100) if t > 0 else 0.0
    return acc, per_class


def main():
    print("Loading tokenizer …")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    special = [DataConfig.SEEKER_TOKEN, DataConfig.SUPPORTER_TOKEN,
               DataConfig.SUPPORTER_TARGET_TOKEN]
    tokenizer.add_tokens(special)

    print(f"Loading test data: {TEST_PATH}")
    with open(TEST_PATH, "r") as f:
        test_data = json.load(f)
    print(f"  samples = {len(test_data['samples'])}")

    # 수정된 샘플 수 확인
    edited = sum(1 for s in test_data["samples"] if s.get("_edited"))
    print(f"  edited  = {edited}\n")

    dataset = ViolationDataset(test_data, tokenizer, max_length=512, is_test=True)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    # header
    print(f"{'Model':<30s}  {'Acc':>6s}  ", end="")
    for lb in LABELS:
        print(f"{lb:>7s}", end="")
    print()
    print("-" * 82)

    for model_file, desc in MODELS.items():
        model_path = os.path.join("models/outputs", model_file)
        if not os.path.exists(model_path):
            print(f"{desc:<30s}  파일 없음")
            continue

        model = ViolationClassifier("roberta-base", num_labels=6)
        model.roberta.resize_token_embeddings(len(tokenizer))
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)

        acc, pc = evaluate_model(model, loader)
        label_str = "  ".join(f"{pc[lb]:5.1f}%" for lb in LABELS)
        print(f"{desc:<30s}  {acc:5.1f}%   {label_str}")

    print()


if __name__ == "__main__":
    main()
