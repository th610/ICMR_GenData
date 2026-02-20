"""
Table 1. Structural Safety Guarantee under Gold Injection (n=220)
================================================================
Section 5.2 — R1 메인 결과: post-detection leakage 0%

수치 수정: DATA dict만 바꾸면 됨.
실행: python table2_safety.py
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
    "total": 220,
    "detected": 213,
    "missed": 7,                    # FN (detector miss)
    "resolved_after_detection": 213, # corrected + failsafed
    "post_detection_leakage": 0,
    "corrected": 201,               # rewrite로 교정 성공
    "failsafed": 12,                # failsafe 템플릿
    # per-type breakdown
    "per_violation": {
        "v1": {"total": 50, "detected": 48, "corrected": 45, "failsafed":  3},
        "v2": {"total": 60, "detected": 58, "corrected": 55, "failsafed":  3},
        "v3": {"total": 30, "detected": 30, "corrected": 28, "failsafed":  2},
        "v4": {"total": 40, "detected": 38, "corrected": 36, "failsafed":  2},
        "v5": {"total": 40, "detected": 39, "corrected": 37, "failsafed":  2},
    },
}

# ════════════════════════════════════════
# 파생 지표 자동 계산
# ════════════════════════════════════════
def compute(d):
    d['detection_rate'] = d['detected'] / d['total']
    d['miss_rate'] = d['missed'] / d['total']
    d['resolved_rate'] = d['resolved_after_detection'] / d['detected'] if d['detected'] else 0
    d['leakage_rate'] = d['post_detection_leakage'] / d['detected'] if d['detected'] else 0
    d['final_safe_rate'] = d['detected'] / d['total']  # = detection_rate (since post-det leakage=0)
    d['correction_rate'] = d['corrected'] / d['detected'] if d['detected'] else 0
    d['failsafe_rate'] = d['failsafed'] / d['detected'] if d['detected'] else 0
    return d


# ════════════════════════════════════════
# PRINT — 콘솔 테이블
# ════════════════════════════════════════
def print_table(d):
    print("=" * 65)
    print("  Table 1. Structural Safety Guarantee (Gold Injection, n=220)")
    print("=" * 65)
    print(f"  {'Metric':<35s} {'Value':>25s}")
    print(f"  {'-'*35} {'-'*25}")
    print(f"  {'Detected':<35s} {d['detected']}/{d['total']} ({d['detection_rate']:.1%})")
    print(f"  {'Missed (FN)':<35s} {d['missed']}/{d['total']} ({d['miss_rate']:.1%})")
    print(f"  {'Resolved after detection':<35s} {d['resolved_after_detection']}/{d['detected']} ({d['resolved_rate']:.1%})")
    print(f"  {'Post-detection leakage':<35s} {d['post_detection_leakage']}/{d['detected']} ({d['leakage_rate']:.1%})")
    print(f"  {'Final safe rate':<35s} {d['final_safe_rate']:.1%}")
    print()
    print(f"  --- Breakdown (of detected) ---")
    print(f"  {'Corrected (rewrite → Normal)':<35s} {d['corrected']}/{d['detected']} ({d['correction_rate']:.1%})")
    print(f"  {'Failsafed (template)':<35s} {d['failsafed']}/{d['detected']} ({d['failsafe_rate']:.1%})")
    print()
    print(f"  {'Type':<6s} {'Total':>6s} {'Detect':>8s} {'Correct':>8s} {'Failsafe':>8s} {'Det%':>8s}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for k in ['v1','v2','v3','v4','v5']:
        v = d['per_violation'][k]
        dr = v['detected']/v['total'] if v['total'] else 0
        print(f"  {k.upper():<6s} {v['total']:>6d} {v['detected']:>8d} {v['corrected']:>8d} {v['failsafed']:>8d} {dr:>7.1%}")
    print()


# ════════════════════════════════════════
# FIGURE — Safety Funnel (Containment Flow)
# ════════════════════════════════════════
def generate_figure(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    box = dict(boxstyle='round,pad=0.4', edgecolor='#333', linewidth=1.5)

    ax.text(0.5, 5, f'Violation\nSamples\n{d["total"]}', ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(**box, facecolor='#FFCDD2'))

    ax.text(3, 7.5, f'Detected\n{d["detected"]} ({d["detection_rate"]:.1%})', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(**box, facecolor='#BBDEFB'))
    ax.text(3, 2.5, f'Missed\n{d["missed"]} ({d["miss_rate"]:.1%})', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(**box, facecolor='#FFECB3'))

    ax.text(6, 8.5, f'Corrected\n{d["corrected"]} ({d["correction_rate"]:.1%})', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(**box, facecolor='#C8E6C9'))
    ax.text(6, 6.2, f'Failsafe\n{d["failsafed"]} ({d["failsafe_rate"]:.1%})', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(**box, facecolor='#D1C4E9'))

    ax.text(9, 7.5, f'Safe Output\n{d["detected"]} (100%)', ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(**box, facecolor='#A5D6A7'))
    ax.text(9, 2.5, f'Leakage\n{d["post_detection_leakage"]} (0%)', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(**box, facecolor='#E8F5E9'))

    arrow_kw = dict(arrowstyle='->', lw=2, color='#555')
    thin = dict(arrowstyle='->', lw=1.2, color='#AAA')
    ax.annotate('', xy=(1.7, 7.2), xytext=(1.2, 5.5), arrowprops=arrow_kw)
    ax.annotate('', xy=(1.7, 3.0), xytext=(1.2, 4.5), arrowprops=thin)
    ax.annotate('', xy=(4.7, 8.3), xytext=(4.2, 7.7), arrowprops=arrow_kw)
    ax.annotate('', xy=(4.7, 6.5), xytext=(4.2, 7.3), arrowprops=arrow_kw)
    ax.annotate('', xy=(7.7, 7.8), xytext=(7.2, 8.3), arrowprops=arrow_kw)
    ax.annotate('', xy=(7.7, 7.2), xytext=(7.2, 6.5), arrowprops=arrow_kw)

    ax.text(1.5, 8.0, 'RoBERTa\nDetection', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(4.5, 9.2, 'Rewrite\nLoop', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(7.5, 9.2, 'Final\nOutput', ha='center', fontsize=9, style='italic', color='#666')

    fig.tight_layout()
    fig.savefig(OUT / 'fig_table1_safety_funnel.pdf', dpi=300)
    fig.savefig(OUT / 'fig_table1_safety_funnel.png', dpi=300)
    plt.close(fig)
    print(f"[OK] Figure → {OUT}/fig_table1_safety_funnel.pdf/png")


# ════════════════════════════════════════
# SAVE JSON
# ════════════════════════════════════════
def save_json(d):
    out = Path(__file__).parent / 'table1_safety.json'
    with open(out, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"[OK] JSON → {out}")


if __name__ == '__main__':
    d = compute(DATA)
    print_table(d)
    generate_figure(d)
    save_json(d)
