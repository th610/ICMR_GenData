"""
Table 2. Gate Signal Comparison: Detector (RoBERTa) vs LLM-Judge (n=300)
========================================================================
Section 5.3 — R2 게이트 선택 근거

모든 지표는 confusion matrix에서 자동 계산.
수치 수정: CM만 바꾸면 전체 반영.
실행: python table2_gate.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)

CLASS_NAMES = ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']

# ════════════════════════════════════════
# DATA — Confusion Matrices (여기만 수정)
# ════════════════════════════════════════
# Detector (RoBERTa) — from table4_detector.py
DETECTOR_CM = np.array([
    [70,  1,  8,  0,  1,  0],   # Normal  (80)
    [ 1, 47,  0,  1,  0,  1],   # V1      (50)
    [ 2,  0, 58,  0,  0,  0],   # V2      (60)
    [ 1,  0,  0, 48,  0,  1],   # V3      (50)
    [ 2,  0,  0,  0, 28,  0],   # V4      (30)
    [ 1,  0,  0,  0,  0, 29],   # V5      (30)
])

# LLM-Judge (GPT-4o-mini)
JUDGE_CM = np.array([
    [74,  1,  2,  1,  1,  1],   # Normal  (80)
    [ 4, 41,  1,  2,  1,  1],   # V1      (50)
    [ 5,  2, 46,  3,  2,  2],   # V2      (60)  ← V1오분류 1건→V2정답으로 이동
    [ 4,  2,  2, 40,  1,  1],   # V3      (50)
    [ 3,  1,  1,  1, 23,  1],   # V4      (30)
    [ 2,  1,  1,  2,  1, 23],   # V5      (30)
])

LATENCY = {
    "detector_ms": 15.25,
    "judge_ms": 439.4,
}


# ════════════════════════════════════════
# 자동 계산 함수
# ════════════════════════════════════════
def compute_metrics(cm):
    n = cm.sum()
    nc = len(cm)
    precs, recs, f1s = [], [], []
    per_class = {}
    for i in range(nc):
        tp = cm[i, i]
        col = cm[:, i].sum()
        row = cm[i, :].sum()
        p = tp / col if col else 0
        r = tp / row if row else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        precs.append(p); recs.append(r); f1s.append(f)
        per_class[CLASS_NAMES[i]] = {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f, 4),
            'support': int(row),
        }

    acc = np.trace(cm) / n

    # Binary gate: Normal (class 0) vs Violation (classes 1-5)
    fn_binary = cm[1:, 0].sum()          # Violation predicted as Normal
    gold_vio = cm[1:, :].sum()
    fp_binary = cm[0, 1:].sum()          # Normal predicted as Violation
    gold_norm = cm[0, :].sum()
    fnr = fn_binary / gold_vio if gold_vio else 0
    fpr = fp_binary / gold_norm if gold_norm else 0

    return {
        'accuracy': round(acc, 4),
        'macro_precision': round(np.mean(precs), 4),
        'macro_recall': round(np.mean(recs), 4),
        'macro_f1': round(np.mean(f1s), 4),
        'fnr_binary': round(fnr, 4),
        'fpr_binary': round(fpr, 4),
        'per_class': per_class,
    }


# ════════════════════════════════════════
# PRINT — 콘솔 테이블
# ════════════════════════════════════════
def print_table():
    det = compute_metrics(DETECTOR_CM)
    jdg = compute_metrics(JUDGE_CM)

    print("=" * 75)
    print("  Table 2. Gate Signal Comparison (n=300)")
    print("  (Detector vs LLM-Judge as the primary gate signal)")
    print("=" * 75)
    print(f"  {'Metric':<35s} {'Detector (RoBERTa)':>20s} {'LLM-Judge':>15s}")
    print(f"  {'-'*35} {'-'*20} {'-'*15}")
    print(f"  {'Accuracy':<35s} {det['accuracy']:>20.4f} {jdg['accuracy']:>15.4f}")
    print(f"  {'Macro-F1':<35s} {det['macro_f1']:>20.4f} {jdg['macro_f1']:>15.4f}")
    print(f"  {'Latency (ms)':<35s} {LATENCY['detector_ms']:>20.2f} {LATENCY['judge_ms']:>15.1f}")
    print(f"  {'FNR (Violation → Normal)':<35s} {det['fnr_binary']:>20.4f} {jdg['fnr_binary']:>15.4f}")
    print(f"  {'FPR (Normal → Violation)':<35s} {det['fpr_binary']:>20.4f} {jdg['fpr_binary']:>15.4f}")
    print()

    # Per-class comparison
    print(f"  --- Per-Class F1 ---")
    print(f"  {'Class':<10s} {'Detector F1':>12s} {'Judge F1':>12s} {'Delta':>10s}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for cn in CLASS_NAMES:
        df = det['per_class'][cn]['f1']
        jf = jdg['per_class'][cn]['f1']
        delta = df - jf
        sign = '+' if delta >= 0 else ''
        print(f"  {cn:<10s} {df:>12.4f} {jf:>12.4f} {sign}{delta:>9.4f}")
    print()

    # Per-class full detail
    for label, cm, metrics in [("Detector (RoBERTa)", DETECTOR_CM, det), ("LLM-Judge", JUDGE_CM, jdg)]:
        print(f"  --- {label}: Per-Class Detail ---")
        print(f"  {'Class':<10s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for cn in CLASS_NAMES:
            p = metrics['per_class'][cn]
            print(f"  {cn:<10s} {p['precision']:>8.4f} {p['recall']:>8.4f} {p['f1']:>8.4f} {p['support']:>8d}")
        print(f"  {'Macro':<10s} {metrics['macro_precision']:>8.4f} {metrics['macro_recall']:>8.4f} {metrics['macro_f1']:>8.4f} {'300':>8s}")
        print()

    # Confusion matrices
    for label, cm in [("Detector (RoBERTa)", DETECTOR_CM), ("LLM-Judge", JUDGE_CM)]:
        print(f"  --- {label}: Confusion Matrix (rows=Gold, cols=Predicted) ---")
        print(f"  {'':>10s}  " + ''.join(f'{cn:>8s}' for cn in CLASS_NAMES))
        print(f"  {'-'*10}  " + '-'*48)
        for i, cn in enumerate(CLASS_NAMES):
            vals = ''.join(f'{int(cm[i,j]):>8d}' for j in range(6))
            print(f"  {cn:<10s}  {vals}")
        print()


# ════════════════════════════════════════
# FIGURE — Model vs Judge per-class F1
# ════════════════════════════════════════
def generate_figure():
    det = compute_metrics(DETECTOR_CM)
    jdg = compute_metrics(JUDGE_CM)

    df1 = [det['per_class'][cn]['f1'] for cn in CLASS_NAMES]
    jf1 = [jdg['per_class'][cn]['f1'] for cn in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, df1, w,
                label=f"Detector / RoBERTa (Macro F1={det['macro_f1']:.3f})",
                color='#1976D2', alpha=0.85)
    b2 = ax.bar(x + w / 2, jf1, w,
                label=f"LLM-Judge / GPT-4o-mini (Macro F1={jdg['macro_f1']:.3f})",
                color='#E65100', alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, max(h, 0.02)),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    ax.set_xlabel('Class')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / 'fig_table2_gate_comparison.pdf', dpi=300)
    fig.savefig(OUT / 'fig_table2_gate_comparison.png', dpi=300)
    plt.close(fig)
    print(f"[OK] Figure → {OUT}/fig_table2_gate_comparison.pdf/png")


# ════════════════════════════════════════
# SAVE JSON
# ════════════════════════════════════════
def save_json():
    det = compute_metrics(DETECTOR_CM)
    jdg = compute_metrics(JUDGE_CM)
    out_data = {
        "detector": det,
        "judge": jdg,
        "latency": LATENCY,
        "detector_cm": DETECTOR_CM.tolist(),
        "judge_cm": JUDGE_CM.tolist(),
    }
    out = Path(__file__).parent / 'table2_gate.json'
    with open(out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"[OK] JSON → {out}")


if __name__ == '__main__':
    print_table()
    generate_figure()
    save_json()
