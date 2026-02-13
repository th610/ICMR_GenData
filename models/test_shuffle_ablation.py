"""
Shuffled Prefix Ablation Study
================================
prefix를 셔플한 상태에서 gen1~3, response(gen4)를 단계별로 추가하며 비교

실험 구성 (6개):
  [Original prefix]
    A1. prefix only
    A2. prefix + gen1~3
    A3. prefix + gen1~3 + gen4 (full)

  [Shuffled prefix]
    B1. shuffled prefix only
    B2. shuffled prefix + gen1~3
    B3. shuffled prefix + gen1~3 + gen4 (full)
"""

import json, random, copy, torch
import numpy as np
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from model import ViolationClassifier
from data_utils import DataConfig, format_conversation_with_truncation

LABELS = DataConfig.LABELS
LABEL2ID = DataConfig.LABEL2ID
MODEL_PATH = "outputs/best_model_v2.pt"
TEST_PATH  = "data/test_gold_300.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset that accepts pre-processed samples ──────────────────────────
class AblationDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prefix   = s["prefix_dialog"]
        gen_dial = s["generated_dialog"]

        text = format_conversation_with_truncation(
            prefix, gen_dial, self.tokenizer, self.max_length
        )
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        label = LABEL2ID[s["primary_label"]]
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── helpers ──────────────────────────────────────────────────────────────
def make_dummy_turn(speaker="seeker"):
    return {"speaker": speaker, "content": ""}

def build_variants(raw_samples):
    """6가지 실험 변형을 만든다."""
    # 셔플용 prefix pool (같은 시드로 재현)
    random.seed(42)
    all_prefixes = [s.get("prefix_dialog", []) for s in raw_samples]
    shuffled_prefixes = all_prefixes.copy()
    random.shuffle(shuffled_prefixes)

    variants = {
        "A1_orig_prefix_only":      [],
        "A2_orig_prefix+gen1-3":    [],
        "A3_orig_full":             [],
        "B1_shuf_prefix_only":      [],
        "B2_shuf_prefix+gen1-3":    [],
        "B3_shuf_full":             [],
    }

    for i, sample in enumerate(raw_samples):
        prefix_orig = sample.get("prefix_dialog", [])
        prefix_shuf = shuffled_prefixes[i]

        if "generated_turn" in sample:
            import json as _json
            gd = _json.loads(sample["generated_turn"])["dialog"]
            prefix_orig = sample.get("prefix_conversation", [])
            prefix_shuf_i = shuffled_prefixes[i]
        else:
            gd = sample["generated_dialog"]
            prefix_shuf_i = prefix_shuf

        gen1, gen2, gen3, gen4 = gd[0], gd[1], gd[2], gd[3]
        label = sample["primary_label"]

        dummy_seeker    = make_dummy_turn("seeker")
        dummy_supporter = make_dummy_turn("supporter")

        # ── A1: original prefix only (gen1~4 = empty) ──
        variants["A1_orig_prefix_only"].append({
            "prefix_dialog":    prefix_orig,
            "generated_dialog": [dummy_seeker, dummy_supporter, dummy_seeker, dummy_supporter],
            "primary_label":    label,
        })

        # ── A2: original prefix + gen1~3 (gen4 = empty) ──
        variants["A2_orig_prefix+gen1-3"].append({
            "prefix_dialog":    prefix_orig,
            "generated_dialog": [gen1, gen2, gen3, dummy_supporter],
            "primary_label":    label,
        })

        # ── A3: original full ──
        variants["A3_orig_full"].append({
            "prefix_dialog":    prefix_orig,
            "generated_dialog": [gen1, gen2, gen3, gen4],
            "primary_label":    label,
        })

        # ── B1: shuffled prefix only ──
        variants["B1_shuf_prefix_only"].append({
            "prefix_dialog":    prefix_shuf_i,
            "generated_dialog": [dummy_seeker, dummy_supporter, dummy_seeker, dummy_supporter],
            "primary_label":    label,
        })

        # ── B2: shuffled prefix + gen1~3 ──
        variants["B2_shuf_prefix+gen1-3"].append({
            "prefix_dialog":    prefix_shuf_i,
            "generated_dialog": [gen1, gen2, gen3, dummy_supporter],
            "primary_label":    label,
        })

        # ── B3: shuffled full ──
        variants["B3_shuf_full"].append({
            "prefix_dialog":    prefix_shuf_i,
            "generated_dialog": [gen1, gen2, gen3, gen4],
            "primary_label":    label,
        })

    return variants


def evaluate(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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

    # overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total   = len(all_labels)
    acc     = correct / total * 100

    # per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    per_class = {}
    for label_id in range(len(LABELS)):
        t = class_total.get(label_id, 0)
        c = class_correct.get(label_id, 0)
        per_class[LABELS[label_id]] = (c / t * 100) if t > 0 else 0.0

    return acc, per_class


# ── main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading tokenizer & model …")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    special = [DataConfig.SEEKER_TOKEN, DataConfig.SUPPORTER_TOKEN,
               DataConfig.SUPPORTER_TARGET_TOKEN]
    tokenizer.add_tokens(special)

    model = ViolationClassifier("roberta-base", num_labels=6)
    model.roberta.resize_token_embeddings(len(tokenizer))
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Loading test data …")
    with open(TEST_PATH, "r") as f:
        test_data = json.load(f)
    raw_samples = test_data["samples"]
    print(f"  samples = {len(raw_samples)}")

    print("Building 6 variants …\n")
    variants = build_variants(raw_samples)

    # header
    print(f"{'Experiment':<30s}  {'Acc':>6s}  ", end="")
    for lb in LABELS:
        print(f"{lb:>7s}", end="")
    print()
    print("-" * 80)

    results = {}
    for name in [
        "A1_orig_prefix_only",
        "A2_orig_prefix+gen1-3",
        "A3_orig_full",
        "B1_shuf_prefix_only",
        "B2_shuf_prefix+gen1-3",
        "B3_shuf_full",
    ]:
        ds  = AblationDataset(variants[name], tokenizer)
        acc, pc = evaluate(model, ds)
        results[name] = {"accuracy": acc, "per_class": pc}

        print(f"{name:<30s}  {acc:5.1f}%  ", end="")
        for lb in LABELS:
            print(f"{pc[lb]:6.1f}%", end="")
        print()

    # summary
    print()
    print("=" * 80)
    print("Summary  (prefix 기여도 = Original − Shuffled)")
    print("=" * 80)
    for pair in [("A1", "B1", "prefix only"),
                 ("A2", "B2", "prefix+gen1-3"),
                 ("A3", "B3", "full")]:
        a_key = [k for k in results if k.startswith(pair[0])][0]
        b_key = [k for k in results if k.startswith(pair[1])][0]
        diff = results[a_key]["accuracy"] - results[b_key]["accuracy"]
        print(f"  {pair[2]:<20s}:  {results[a_key]['accuracy']:5.1f}%  → {results[b_key]['accuracy']:5.1f}%  (Δ = {diff:+.1f}%p)")

    print()
    print("gen1~3 기여도 (셔플 기준):  "
          f"{results['B2_shuf_prefix+gen1-3']['accuracy']:5.1f}% − "
          f"{results['B1_shuf_prefix_only']['accuracy']:5.1f}% = "
          f"{results['B2_shuf_prefix+gen1-3']['accuracy'] - results['B1_shuf_prefix_only']['accuracy']:+.1f}%p")

    print("response(gen4) 기여도 (셔플 기준):  "
          f"{results['B3_shuf_full']['accuracy']:5.1f}% − "
          f"{results['B2_shuf_prefix+gen1-3']['accuracy']:5.1f}% = "
          f"{results['B3_shuf_full']['accuracy'] - results['B2_shuf_prefix+gen1-3']['accuracy']:+.1f}%p")

    print("response(gen4) 기여도 (원본 기준):  "
          f"{results['A3_orig_full']['accuracy']:5.1f}% − "
          f"{results['A2_orig_prefix+gen1-3']['accuracy']:5.1f}% = "
          f"{results['A3_orig_full']['accuracy'] - results['A2_orig_prefix+gen1-3']['accuracy']:+.1f}%p")


if __name__ == "__main__":
    main()
