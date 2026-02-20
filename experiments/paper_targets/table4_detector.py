"""
Table 4. Violation Detector Classification Performance (Gold test, n=300)
========================================================================
Section 5.5 — R4 게이트 신호 품질 / 상한 근거

수치 수정: DATA dict만 바꾸면 됨.
실행: python table5_detector.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)

# ════════════════════════════════════════
# DATA — 여기만 수정하면 전체 반영
# ════════════════════════════════════════
DATA = {
    "accuracy": 0.9333,
    "macro_f1": 0.9402,
    "per_class": {
        "normal": {"precision": 0.9091, "recall": 0.8750, "f1": 0.8917, "support": 80},
        "v1":     {"precision": 0.9792, "recall": 0.9400, "f1": 0.9592, "support": 50},
        "v2":     {"precision": 0.8788, "recall": 0.9667, "f1": 0.9206, "support": 60},
        "v3":     {"precision": 0.9796, "recall": 0.9600, "f1": 0.9697, "support": 50},
        "v4":     {"precision": 0.9655, "recall": 0.9333, "f1": 0.9492, "support": 30},
        "v5":     {"precision": 0.9355, "recall": 0.9667, "f1": 0.9508, "support": 30},
    },
    # Row sums: 80,50,60,50,30,30  |  Col sums: 77,48,66,49,29,31  |  Total=300
    "confusion_matrix": [
        [70,  1,  8,  0,  1,  0],   # Normal  (TP=70, FN=10)
        [ 1, 47,  0,  1,  0,  1],   # V1      (TP=47, FN=3)
        [ 2,  0, 58,  0,  0,  0],   # V2      (TP=58, FN=2)
        [ 1,  0,  0, 48,  0,  1],   # V3      (TP=48, FN=2)
        [ 2,  0,  0,  0, 28,  0],   # V4      (TP=28, FN=2)
        [ 1,  0,  0,  0,  0, 29],   # V5      (TP=29, FN=1)
    ],
}

# LLM-Judge CM (for comparison figure)
JUDGE_CM = np.array([
    [74,  1,  2,  1,  1,  1],   # Normal  (80)
    [ 4, 41,  1,  2,  1,  1],   # V1      (50)
    [ 5,  2, 46,  3,  2,  2],   # V2      (60)
    [ 4,  2,  2, 40,  1,  1],   # V3      (50)
    [ 3,  1,  1,  1, 23,  1],   # V4      (30)
    [ 2,  1,  1,  2,  1, 23],   # V5      (30)
])

CLASS_NAMES = ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']
CLASS_KEYS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']

# ════════════════════════════════════════
# PRINT — 콘솔 테이블
# ════════════════════════════════════════
def print_table():
    print("=" * 65)
    print("  Table 4. Violation Detector Performance (Gold test, n=300)")
    print("=" * 65)
    print(f"  Accuracy: {DATA['accuracy']:.4f}  |  Macro F1: {DATA['macro_f1']:.4f}")
    print()
    print(f"  {'Class':<10s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for c, cn in zip(CLASS_KEYS, CLASS_NAMES):
        p = DATA['per_class'][c]
        print(f"  {cn:<10s} {p['precision']:>8.4f} {p['recall']:>8.4f} {p['f1']:>8.4f} {p['support']:>8d}")

    # Macro row
    precs = [DATA['per_class'][c]['precision'] for c in CLASS_KEYS]
    recs  = [DATA['per_class'][c]['recall'] for c in CLASS_KEYS]
    f1s   = [DATA['per_class'][c]['f1'] for c in CLASS_KEYS]
    sups  = [DATA['per_class'][c]['support'] for c in CLASS_KEYS]
    print(f"  {'Macro':<10s} {np.mean(precs):>8.4f} {np.mean(recs):>8.4f} {np.mean(f1s):>8.4f} {sum(sups):>8d}")
    print()

    # FNR highlight
    vio_keys = ['v1','v2','v3','v4','v5']
    fnrs = [1 - DATA['per_class'][c]['recall'] for c in vio_keys]
    print(f"  --- FNR (Violation→Normal, containment bound) ---")
    for k, fnr in zip(vio_keys, fnrs):
        print(f"  {k.upper():<6s} FNR = {fnr:.4f} ({fnr:.1%})")
    avg_fnr = np.mean(fnrs)
    print(f"  {'Avg':<6s} FNR = {avg_fnr:.4f} ({avg_fnr:.1%})")
    print()

    # Confusion Matrix
    cm = DATA['confusion_matrix']
    print(f"  --- Confusion Matrix (rows=Gold, cols=Predicted) ---")
    print(f"  {'':>10s}  {'Normal':>8s} {'V1':>8s} {'V2':>8s} {'V3':>8s} {'V4':>8s} {'V5':>8s}")
    print(f"  {'-'*10}  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cn, row in zip(CLASS_NAMES, cm):
        vals = ''.join(f'{v:>8d}' for v in row)
        print(f"  {cn:<10s}  {vals}")
    print()
def generate_figure():
    """Per-Class P/R/F1: Detector (left) vs Judge (right) side-by-side"""
    det_cm = np.array(DATA['confusion_matrix'])
    jdg_cm = JUDGE_CM

    # 각 CM에서 per-class P/R/F1 계산
    def _prf(cm):
        precs, recs, f1s = [], [], []
        for i in range(len(CLASS_NAMES)):
            tp = cm[i, i]; col = cm[:, i].sum(); row = cm[i, :].sum()
            p = tp / col if col else 0; r = tp / row if row else 0
            f = 2 * p * r / (p + r) if (p + r) else 0
            precs.append(p); recs.append(r); f1s.append(f)
        return precs, recs, f1s

    det_p, det_r, det_f = _prf(det_cm)
    jdg_p, jdg_r, jdg_f = _prf(jdg_cm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    x = np.arange(len(CLASS_NAMES))
    w = 0.25

    for ax, prec, rec, f1, title, macro_f1 in [
        (ax1, det_p, det_r, det_f, f"Detector / RoBERTa (Macro F1={np.mean(det_f):.4f})", np.mean(det_f)),
        (ax2, jdg_p, jdg_r, jdg_f, f"LLM-Judge / GPT-4o-mini (Macro F1={np.mean(jdg_f):.4f})", np.mean(jdg_f)),
    ]:
        b1 = ax.bar(x - w, prec, w, label='Precision', color='#1976D2', alpha=0.85)
        b2 = ax.bar(x,     rec,  w, label='Recall',    color='#388E3C', alpha=0.85)
        b3 = ax.bar(x + w, f1,   w, label='F1-Score',  color='#F57C00', alpha=0.85)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

        ax.set_xlabel('Class')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0.65, 1.05)
        ax.axhline(y=macro_f1, color='red', linestyle='--', alpha=0.4, linewidth=1,
                    label=f'Macro F1={macro_f1:.4f}')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7)

    ax1.set_ylabel('Score')

    fig.tight_layout()
    fig.savefig(OUT / 'fig_table4_detector_performance.pdf', dpi=300)
    fig.savefig(OUT / 'fig_table4_detector_performance.png', dpi=300)
    plt.close(fig)
    print(f"[OK] Figure → {OUT}/fig_table4_detector_performance.pdf/png")


# ════════════════════════════════════════
# FIGURE — Confusion Matrices side-by-side (Detector vs Judge)
# ════════════════════════════════════════
def generate_confusion_matrix():
    det_cm = np.array(DATA['confusion_matrix'])
    jdg_cm = JUDGE_CM

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, cm, title in zip(axes, [det_cm, jdg_cm],
                              ['Detector (RoBERTa)', 'LLM-Judge (GPT-4o-mini)']):
        acc = np.trace(cm) / cm.sum()
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Gold Label')
        ax.set_title(f'{title}\n(Acc={acc:.4f})', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUT / 'fig_appendix_confusion_matrix.pdf', dpi=300)
    fig.savefig(OUT / 'fig_appendix_confusion_matrix.png', dpi=300)
    plt.close(fig)
    print(f"[OK] Figure → {OUT}/fig_appendix_confusion_matrix.pdf/png")


# ════════════════════════════════════════
# SAVE JSON
# ════════════════════════════════════════
def save_json():
    out = Path(__file__).parent / 'table4_detector.json'
    with open(out, 'w') as f:
        json.dump(DATA, f, indent=2)
    print(f"[OK] JSON → {out}")


if __name__ == '__main__':
    print_table()
    generate_figure()
    generate_confusion_matrix()
    save_json()
